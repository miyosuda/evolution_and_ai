# -*- coding: utf-8 -*-
import numpy as np
import random
import json
import sys
import config
import time
import os
from gym.wrappers import Monitor
np.set_printoptions(precision=3, threshold=20, linewidth=200)

from vae_racing import VAERacing

PEEK_PROB    = 0.3
SIMPLE_MODE  = True

final_mode   = False
render_mode  = True
record_video = False
MEAN_MODE    = False


def passthru(x):
    return x    


class Agent:
    ''' simple feedforward model to act on world model's hidden state '''

    def __init__(self,
                 layer_1=10,            # 10
                 layer_2=5,             # 0
                 input_size=5 + 20 * 2, # 26
                 output_size=1):        # 3

        self.layer_1 = layer_1
        self.layer_2 = layer_2
        
        self.input_size  = input_size   # 
        self.output_size = output_size  # action space
        
        if layer_2 == 0:
            self.shapes = [(self.input_size, self.layer_1),
                           (self.layer_1,    self.output_size)]
                            
        else:
            self.shapes = [(self.input_size, self.layer_1),
                           (self.layer_1,    self.layer_2),
                           (self.layer_2,    self.output_size)]
                            

        self.activations = [
            np.tanh, np.tanh, np.tanh
        ]  # assumption that output is bounded between -1 and 1 (pls chk!)

        self.weight = []
        self.bias = []
        self.param_count = 0

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

    def get_action(self, x):
        h = np.array(x).flatten()
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i] # (16,10)
            b = self.bias[i]   # (10)
            h = np.matmul(h, w) + b
            h = self.activations[i](h)
        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev


def _clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


class SimpleWorldModel:
    def __init__(self, obs_size=16, action_size=3, hidden_size=10):
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.shapes = [(self.obs_size + self.action_size, self.hidden_size),
                       (self.hidden_size, self.obs_size)]

        self.weight = []
        self.bias = []
        self.param_count = 0

        self.dt = 1.0 / 50.0  # 50 fps

        self.hidden_state = np.zeros(self.hidden_size)

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

    def reset(self):
        self.hidden_state = np.zeros(self.hidden_size)

    def predict_next_obs(self, obs, action):
        obs = np.array(obs).flatten()

        new_action = np.array([0.0, 0.0, 0.0])

        new_action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
        new_action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
        new_action[1] = (action[1] + 1.0) / 2.0
        new_action[2] = _clip(action[2])

        h = np.concatenate([obs, new_action.flatten()])

        activations = [np.tanh, passthru]

        num_layers = 2

        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            h = activations[i](h)
            if (i == 0):  # save the hidden state
                self.hidden_state = h.flatten()

        prediction = obs + h.flatten() * self.dt  # residual

        return prediction

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev


class CustomModel:
    ''' learning the best feed forward model for vae_racing '''

    def __init__(self, game):
        self.output_noise = game.output_noise
        self.env_name = game.env_name
        self.world_hidden_size = game.layers[0]
        self.agent_hidden_size = game.layers[1]
        self.rnn_mode = False  # in the future will be useful
        self.experimental_mode = True
        self.peek_prob = PEEK_PROB
        self.simple_mode = SIMPLE_MODE
        self.peek_next = 1
        self.peek = 1
        self.counter = 0

        self.input_size = game.input_size  # observation size
        self.output_size = game.output_size  # action size

        self.render_mode = False

        self.world_model = SimpleWorldModel(
            obs_size=self.input_size,
            action_size=self.output_size,
            hidden_size=self.world_hidden_size)
        
        agent_input_size = self.input_size + self.world_hidden_size
        
        if self.simple_mode:
            agent_input_size = self.input_size
            
        self.agent = Agent(
            layer_1=self.agent_hidden_size,
            layer_2=0,
            input_size=agent_input_size,
            output_size=self.output_size)

        self.param_count = self.world_model.param_count + self.agent.param_count
        self.prev_action = np.zeros(self.output_size)
        self.prev_prediction = None

    def reset(self):
        self.prev_prediction = None
        self.peek_next = 1
        self.peek = 1
        self.counter = 0
        self.world_model.reset()

    def make_env(self, seed=-1, render_mode=False):
        self.render_mode = render_mode

        self.env = VAERacing()
        if (seed >= 0):
            self.env.seed(seed)

    def get_action(self, real_obs, t=0, mean_mode=False):
        obs = real_obs
        use_prediction = False

        self.counter += 1  # for tracking frames in case we want to dump out rgb images

        if (self.prev_prediction is not None) and (self.peek_next == 0):
            obs = self.prev_prediction
            use_prediction = True

        if self.simple_mode:
            agent_obs = obs
        else:
            prev_hidden = self.world_model.hidden_state.flatten()
            agent_obs = np.concatenate(
                [obs.flatten(), prev_hidden])  # use previous hidden state
        action = self.agent.get_action(agent_obs)

        self.peek = self.peek_next
        self.peek_next = 0
        if (np.random.rand() < self.peek_prob):
            self.peek_next = 1

        self.prev_prediction = self.world_model.predict_next_obs(obs, action)
        # update hidden state, and predict next frame
        return action

    def set_model_params(self, model_params):
        world_params = model_params[:self.world_model.param_count]
        agent_params = model_params[self.world_model.param_count:
                                    self.world_model.param_count +
                                    self.agent.param_count]

        assert len(world_params) == self.world_model.param_count, \
            "inconsistent world model params"

        assert len(agent_params) == self.agent.param_count, "inconsistent agent params"
        
        self.world_model.set_model_params(world_params)
        self.agent.set_model_params(agent_params)

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev


def evaluate(model):
    # run 100 times and average score, according to the reles.
    model.env.seed(0)
    total_reward = 0.0
    N = 100
    for i in range(N):
        reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
        total_reward += reward[0]
    return (total_reward / float(N))


def simulate(model,
             train_mode=False,
             render_mode=True,
             num_episode=5,
             seed=-1,
             max_len=-1):

    reward_list = []
    t_list = []

    max_episode_length = 1000

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    for episode in range(num_episode):
        
        if model.rnn_mode:
            model.reset()

        if model.experimental_mode:
            model.reset()

        obs = model.env.reset()
        
        if obs is None:
            obs = np.zeros(model.input_size)

        total_reward = 0.0
        reward_threshold = 300  # consider we have won if we got more than this

        num_glimpse = 0

        for t in range(max_episode_length):
            if render_mode:
                model.env.render("human")

            if model.rnn_mode:
                action = model.get_action(obs)
            else:
                if MEAN_MODE:
                    action = model.get_action(obs, t=t, mean_mode=(not train_mode))
                else:
                    action = model.get_action(obs, t=t, mean_mode=False)

            prev_obs = obs

            obs, reward, done, info = model.env.step(action)

            if model.experimental_mode:  # augment reward with prob
                num_glimpse += model.peek

            total_reward += reward

            if done:
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)
            if model.experimental_mode:
                print("percent glimpse", float(num_glimpse) / float(t + 1.0))
        reward_list.append(total_reward)
        t_list.append(t)

    return reward_list, t_list


def main():
    assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

    gamename = sys.argv[1]

    use_model = False

    game = config.games[gamename]

    if len(sys.argv) > 2:
        use_model = True
        filename = sys.argv[2]

    the_seed = 0
    if len(sys.argv) > 3:
        the_seed = int(sys.argv[3])

    model = CustomModel(game)

    model.make_env(render_mode=render_mode)

    if use_model:
        model.load_model(filename)
    else:
        params = model.get_random_model_params(stdev=0.5)
        model.set_model_params(params)

    if final_mode:
        rewards = []

        for i in range(100):
            reward, steps_taken = simulate(model,
                                           train_mode=False,
                                           render_mode=False,
                                           num_episode=1,
                                           seed=the_seed + i)
            print(i, reward)
            rewards.append(reward[0])
        print("seed", the_seed,
              "average_reward", np.mean(rewards),
              "standard_deviation", np.std(rewards))
    else:
        if record_video:
            model.env = Monitor(
                model.env,
                directory='/tmp/' + gamename,
                video_callable=lambda episode_id: True,
                write_upon_reset=True,
                force=True)
        for i in range(1):
            reward, steps_taken = simulate(
                model,
                train_mode=False,
                render_mode=render_mode,
                num_episode=1,
                seed=the_seed + i)
            print("terminal reward", reward, "average steps taken",
                  np.mean(steps_taken) + 1)


if __name__ == "__main__":
    main()
