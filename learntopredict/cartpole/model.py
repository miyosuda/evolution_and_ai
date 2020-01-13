# -*- coding: utf-8 -*-
import numpy as np
import random
# I implemented Schmidhuber's "Compressed Network Search" but didn't use it.
# ndded for the compress/decompress functions.
#from scipy.fftpack import dct
import json
import sys
import config
from env import make_env
import time
import os

#import cma
#from es import SimpleGA, CMAES, PEPG, OpenES

from gym.wrappers import Monitor

final_mode = False
render_mode = True
RENDER_DELAY = True
record_video = False
MEAN_MODE = False
record_rgb = False

if record_rgb:
    import imageio


def compress_2d(w, shape=None):
    s = w.shape
    if shape:
        s = shape
    c = dct(dct(w, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')
    return c[0:s[0], 0:s[1]]


def decompress_2d(c, shape):
    c_out = np.zeros(shape)
    c_out[0:c.shape[0], 0:c.shape[1]] = c
    w = dct(dct(c_out.T, type=3, norm='ortho').T, type=3, norm='ortho')
    return w


def compress_1d(w, shape=None, axis=0):
    s = w.shape
    if shape:
        s = shape
    c = dct(w, axis=axis, type=2, norm='ortho')
    return c[0:s[0], 0:s[1]]


def decompress_1d(c, shape, axis=0):
    c_out = np.zeros(shape)
    c_out[0:c.shape[0], 0:c.shape[1]] = c
    w = dct(c_out, axis=axis, type=3, norm='ortho')
    return w


def make_model(game):
    # can be extended in the future.
    if game.experimental_mode:
        # こちらを通っている.
        model = CustomModel(game)
    else:
        model = Model(game)
    return model


def sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0


def relu(x):
    return np.maximum(x, 0)


def passthru(x):
    return x


# useful for discrete actions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# useful for discrete actions
def sample(p):
    return np.argmax(np.random.multinomial(1, p))


class SimpleWorldModel:
    ''' deterministic LSTM model for cart-pole swing up task '''

    def __init__(self, obs_size=5, action_size=1, hidden_size=80):
        self.obs_size = obs_size  # x, cos(theta), sin(theta).
        self.action_size = action_size  # between -1 and 1
        self.reward_size = 1  # reward
        self.hidden_size = hidden_size

        self.x_threshold = 2.4

        self.hard_reward = True  # reward = (np.cos(theta)+1.0)/2.0 if True

        if self.hard_reward:
            self.shapes = [
                (self.obs_size - 1 + self.action_size, self.hidden_size),
                # input layer, but theta -> cos,sin (so + 1)
                #(self.hidden_size, self.hidden_size), # hidden layer 1
                #(self.hidden_size, self.hidden_size), # hidden layer 2
                (self.hidden_size, 2)
            ]  # output x_dot, and theta_dot
        else:
            self.shapes = [
                (self.obs_size - 1 + self.action_size,
                 self.hidden_size),  # input layer
                #(self.hidden_size, self.hidden_size), # hidden layer 1
                #(self.hidden_size, self.hidden_size), # hidden layer 2
                (self.hidden_size, 2),  # output layer
                (self.obs_size, self.reward_size)
            ]  # predict rewards

        self.weight = []
        self.bias = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            idx += 1

    def state2obs(self, state):
        [x, x_dot, theta, theta_dot] = state
        c = np.cos(theta)
        s = np.sin(theta)
        return [x, x_dot, c, s, theta_dot]

    def predict_dynamics(self, obs, action):
        [x, x_dot, c, s, theta_dot] = obs

        h = np.concatenate([[x_dot, c, s, theta_dot],
                            np.array(action).flatten()])

        activations = [np.tanh, passthru]

        num_layers = 2

        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            h = activations[i](h)

        [x_dot_dot, theta_dot_dot] = h

        return x_dot_dot, theta_dot_dot

    def predict_reward(self, current_obs):
        if self.hard_reward:
            [x, x_dot, c, s, theta_dot] = current_obs
            reward_theta = (c + 1.0) / 2.0
            reward_x = np.cos((x / self.x_threshold) * (np.pi / 2.0))
            reward = reward_theta * reward_x
            return reward
        else:
            h = np.array(current_obs).flatten()
            w = self.weight[4]
            b = self.bias[4]
            reward = np.tanh(np.matmul(h, w) + b)  # linear reward
            return reward.flatten()[0]

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
    ''' learning the best feed forward model for cartpole-swingup '''

    def __init__(self, game):
        self.env_name = game.env_name

        self.layer_1 = game.layers[0]
        self.layer_2 = game.layers[1]
        self.world_hidden_size = self.layer_1
        self.agent_hidden_size = self.layer_2
        self.x_threshold = 2.4
        self.dt = 0.01
        self.peek_prob = 0
        with open("peek_prob.json") as f:
            self.peek_prob = json.load(f)
        self.peek_next = 1
        self.peek = 1

        self.rnn_mode = False
        self.experimental_mode = game.experimental_mode

        self.input_size = game.input_size
        self.output_size = game.output_size

        self.render_mode = False

        self.world_model = SimpleWorldModel(
            obs_size=self.input_size,
            action_size=self.output_size,
            hidden_size=self.world_hidden_size)

        self.agent = Agent(
            layer_1=self.agent_hidden_size,
            layer_2=0,
            input_size=self.input_size,
            output_size=1)

        self.param_count = self.world_model.param_count + self.agent.param_count

    def reset(self):
        self.prev_prediction = None
        self.peek_next = 1
        self.peek = 1

    def make_env(self, seed=-1, render_mode=False):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

    def get_action(self, prev_obs, t=0, mean_mode=False):
        obs = prev_obs  # peek == 1

        if self.prev_prediction != None and (self.peek_next == 0):
            # 観測を予測で置き換え
            obs = self.prev_prediction

        [prev_x, prev_x_dot, prev_c, prev_s, prev_theta_dot] = obs

        all_action = self.agent.get_action(obs)
        # Actionを選択
        action = [all_action[0]]

        self.peek = self.peek_next
        self.peek_next = 0
        if (np.random.rand() < self.peek_prob):
            self.peek_next = 1

        # 選択したActionによる次stateを予測しておく
        prev_theta = np.arctan2(prev_s, prev_c)
        next_x_dot_dot, next_theta_dot_dot = self.world_model.predict_dynamics(
            obs, action)

        next_x = prev_x + prev_x_dot * self.dt
        next_theta = prev_theta + prev_theta_dot * self.dt
        next_x_dot = prev_x_dot + next_x_dot_dot * self.dt
        next_theta_dot = prev_theta_dot + next_theta_dot_dot * self.dt

        next_c = np.cos(next_theta)
        next_s = np.sin(next_theta)

        next_obs = [next_x, next_x_dot, next_c, next_s, next_theta_dot]

        self.prev_prediction = next_obs
        return action

    def set_model_params(self, model_params):
        world_params = model_params[:self.world_model.param_count]
        agent_params = model_params[self.world_model.param_count:
                                    self.world_model.param_count +
                                    self.agent.param_count]

        assert len(world_params) == self.world_model.param_count, \
            "inconsistent world model params"
        assert len(agent_params
                   ) == self.agent.param_count, "inconsistent agent params"

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


class Agent:
    ''' simple feedforward model to act on world model's hidden state '''

    def __init__(self,
                 layer_1=10,
                 layer_2=5,
                 input_size=5 + 20 * 2,
                 output_size=1):
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.input_size = input_size  #
        self.output_size = output_size  # action space
        if layer_2 == 0:
            self.shapes = [(self.input_size, self.layer_1), (self.layer_1,
                                                             self.output_size)]
        else:
            self.shapes = [(self.input_size, self.layer_1),
                           (self.layer_1, self.layer_2), (self.layer_2,
                                                          self.output_size)]

        self.activations = [
            np.tanh, np.tanh, np.tanh
        ]  # assumption that output is bounded between -1 and 1 (pls chk!)

        self.weight = []
        self.bias = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            idx += 1

    def get_action(self, x):
        h = np.array(x).flatten()
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
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


# older model:
"""
class Model:
    ''' simple feedforward model '''

    def __init__(self, game):
        self.output_noise = game.output_noise
        self.env_name = game.env_name
        self.layer_1 = game.layers[0]
        self.layer_2 = game.layers[1]
        self.rnn_mode = False  # in the future will be useful
        self.experimental_mode = False
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = game.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        if game.time_factor > 0:
            self.time_factor = float(game.time_factor)
            self.time_input = 1
        self.input_size = game.input_size
        self.output_size = game.output_size
        if self.layer_2 > 0:
            self.shapes = [(self.input_size + self.time_input, self.layer_1),
                           (self.layer_1, self.layer_2), (self.layer_2,
                                                          self.output_size)]
        elif self.layer_2 == 0:
            self.shapes = [(self.input_size + self.time_input, self.layer_1),
                           (self.layer_1, self.output_size)]
        else:
            assert False, "invalid layer_2"

        self.sample_output = False
        if game.activation == 'relu':
            self.activations = [relu, relu, passthru]
        elif game.activation == 'sigmoid':
            self.activations = [np.tanh, np.tanh, sigmoid]
        elif game.activation == 'softmax':
            self.activations = [np.tanh, np.tanh, softmax]
            self.sample_output = True
        elif game.activation == 'passthru':
            self.activations = [np.tanh, np.tanh, passthru]
        else:
            self.activations = [np.tanh, np.tanh, np.tanh]

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            if self.output_noise[idx]:
                self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)
            idx += 1

        self.render_mode = False

    def make_env(self, seed=-1, render_mode=False):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

    def get_action(self, x, t=0, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            if (self.output_noise[i] and (not mean_mode)):
                out_size = self.shapes[i][1]
                out_std = self.bias_std[i]
                output_noise = np.random.randn(out_size) * out_std
                h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = sample(h)

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
            if self.output_noise[i]:
                s = b_shape
                self.bias_log_std[i] = np.array(
                    model_params[pointer:pointer + s])
                self.bias_std[i] = np.exp(
                    self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
                if self.render_mode:
                    print("bias_std, layer", i, self.bias_std[i])
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
"""


def evaluate(model):
    # run 100 times and average score, according to the reles.
    model.env.seed(0)
    total_reward = 0.0
    N = 1536  #100
    for i in range(N):
        reward, t = simulate(
            model, train_mode=False, render_mode=False, num_episode=1)
        total_reward += reward[0]
    return (total_reward / float(N))


def compress_input_dct(obs):
    new_obs = np.zeros((8, 8))
    for i in range(obs.shape[2]):
        new_obs = +compress_2d(obs[:, :, i] / 255., shape=(8, 8))
    new_obs /= float(obs.shape[2])
    return new_obs.flatten()


def simulate(model,
             train_mode=False,
             render_mode=True,
             num_episode=5,
             seed=-1,
             max_len=-1):

    reward_list = []
    t_list = []

    is_biped = (model.env_name.find("BipedalWalker") >= 0)

    orig_mode = True  # hack for bipedhard's reward augmentation during training (set to false for hack)

    if is_biped:
        orig_mode = False

    dct_compress_mode = False

    max_episode_length = 3000

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    render_flag = "human"
    if record_rgb:
        render_flag = "rgb_array"
        video_path = model.env_name
        if not os.path.exists(video_path):
            os.makedirs(video_path)

    for episode in range(num_episode):
        if model.rnn_mode:
            model.reset()
        if model.experimental_mode:
            model.reset()

        obs = model.env.reset()
        if dct_compress_mode and obs is not None:
            obs = compress_input_dct(obs)

        if obs is None:
            obs = np.zeros(model.input_size)

        total_reward = 0.0
        stumbled = False  # hack for bipedhard's reward augmentation during training. turned off.
        reward_threshold = 300  # consider we have won if we got more than this

        actual_reward = 0
        num_glimpse = 0

        for t in range(max_episode_length):

            if render_mode:
                if model.experimental_mode:
                    if model.peek:
                        render_screen = model.env.render(render_flag)
                    else:
                        [x, x_dot, c, s, theta_dot] = model.prev_prediction
                        theta = np.arctan2(s, c)
                        state = [x, x_dot, theta, theta_dot]
                        # ここでpredictionを表示している
                        render_screen = model.env._render(
                            render_flag, override_state=state)

                else:
                    render_screen = model.env.render(render_flag)

                if RENDER_DELAY:
                    time.sleep(0.01)

                if record_rgb:
                    img = render_screen[300:900, :, :]
                    imageio.imwrite(
                        os.path.join(video_path, (format(t, "05d") + ".png")),
                        img)

            if model.rnn_mode:
                action = model.get_action(obs)
            else:
                if MEAN_MODE:
                    action = model.get_action(
                        obs, t=t, mean_mode=(not train_mode))
                else:
                    action = model.get_action(obs, t=t, mean_mode=False)

            prev_obs = obs

            obs, reward, done, info = model.env.step(action)

            if model.experimental_mode:  # augment reward with prob
                actual_reward += reward
                #reward *= (1 - model.peek) # punish agent for peeking
                num_glimpse += model.peek

            if dct_compress_mode:
                obs = compress_input_dct(obs)

            if train_mode and reward == -100 and (not orig_mode):
                # hack for bipedhard's reward augmentation during training. turned off.
                reward = 0
                stumbled = True

            if (render_mode):
                pass

            total_reward += reward

            if done:
                if train_mode and (not stumbled) and (
                        total_reward > reward_threshold) and (not orig_mode):
                    # hack for bipedhard's reward augmentation during training. turned off.
                    total_reward += 100
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)
            if model.experimental_mode:
                print("actual reward", actual_reward, "percent glimpse",
                      float(num_glimpse) / float(t + 1.0))
        reward_list.append(total_reward)
        t_list.append(t)

    return reward_list, t_list


def main():

    global RENDER_DELAY

    assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

    gamename = sys.argv[1]

    if gamename.startswith("bullet"):
        RENDER_DELAY = True

    use_model = False

    game = config.games[gamename]

    if len(sys.argv) > 2:
        use_model = True
        filename = sys.argv[2]
        print("filename", filename)

    #the_seed = 0
    the_seed = np.random.randint(100000000)

    if len(sys.argv) > 3:
        the_seed = int(sys.argv[3])
        print("seed", the_seed)

    model = make_model(game)
    print('model size', model.param_count)

    model.make_env(render_mode=render_mode)

    if use_model:
        model.load_model(filename)
    else:
        params = model.get_random_model_params(stdev=1.0)
        model.set_model_params(params)

    if final_mode:
        rewards = []

        for i in range(100):  # 6144
            reward, steps_taken = simulate(
                model,
                train_mode=False,
                render_mode=False,
                num_episode=1,
                seed=the_seed + i)
            rewards.append(reward[0])
        print("seed", the_seed, ", average_reward", np.mean(rewards),
              ", standard_deviation", np.std(rewards))
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
