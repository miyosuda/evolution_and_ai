# -*- coding: utf-8 -*-
import numpy as np
import collections

from model import phi, phi_prime, urb_senn_rhs


class PeriodicAccumulator:
    def _get_size(self, key):
        if key == 'y':
            return 3 + 2 * self.n_syn
        elif key in ['g_E_Ds',
                     'syn_pots_sums',
                     'PIVs',
                     'pos_PIVs',
                     'neg_PIVs',
                     'weights',
                     'weight_updates',
                     'deltas',
                     'pre_spikes',
                     'dendr_pred']:
            return self.n_syn
        else:
            return 1

    def __init__(self, keys, interval=1, init_size=1024):
        self.keys      = keys
        self.init_size = init_size
        self.i         = interval
        self.j         = 0
        self.size      = init_size
        self.interval  = interval
        self.t         = np.zeros(init_size, np.float32)

    def prepare_arrays(self, n_syn=1):
        self.n_syn = n_syn
        self.res = {}
        for key in self.keys:
            self.res[key] = np.zeros((self.init_size, self._get_size(key)), np.float32)

    def add(self, curr_t, **vals):
        if np.isclose(self.i, self.interval):
            if self.j == self.size:
                self.t = np.concatenate((self.t, np.zeros(self.t.shape, np.float32)))
                for key in self.keys:
                    self.res[key] = np.vstack((self.res[key],
                                               np.zeros(self.res[key].shape, np.float32)))
                self.size = self.size * 2

            for key in self.keys:
                self.res[key][self.j, :] = np.atleast_2d(vals[key])
            self.t[self.j] = curr_t

            self.j += 1
            self.i = 0
        self.i += 1

    def cleanup(self):
        self.t = self.t[:self.j]
        for key in self.keys:
            self.res[key] = np.squeeze(self.res[key][:self.j, :])

    def add_variable(self, name, val):
        self.res[name] = val


def get_default(params):
    import json
    return json.load(open('./default/default_{0}.json'.format(params), 'r'))


def get_all_save_keys():
    """ all variables that can be recorded during a simulation """
    return ['g_E_Ds',
            'syn_pots_sums',
            'y',
            'spike',
            'dendr_pred',
            'h',
            'PIVs',
            'pos_PIVs',
            'neg_PIVs',
            'dendr_spike',
            'pre_spikes',
            'weights',
            'weight_updates',
            'deltas',
            'I_ext']


def get_fixed_spiker(spikes):
    """
    returns a somatic spiker that implements a fixed spiker, i.e. it will return true
    whenever the current time is listed in the argument spikes
    """
    return lambda curr, dt, **kwargs: spikes.shape[0] > 0 and \
        np.min(np.abs(curr['t'] - spikes)) < dt / 2


def get_phi_U_learner(neuron, dt, factor=1.0):
    """
    returns a dendritic learning signal that has full access to the somatic voltage
        and rate-function phi
    i.e. the average version of inst_backprop if somatic spikes are initiated
        based on an inhomogenuous poisson process with rate phi
    """
    def phi_U_learner(curr, **kwargs):
        # phi(U) * dtを返す
        return factor * phi(curr['y'][0], neuron) * dt
    return phi_U_learner


def step_current(steps):
    """
    defines a step current, i.e. an externally applied current defined by step
    changes, e.g. [][0,0],[2,4],[5,-1]] defines a current that is initially zero,
    stepped to 4 at 2ms, and stepped to -1 at 5ms
    """
    return lambda t: steps[steps[:, 0] <= t, 1][-1]


def run(sim,
        spiker,
        spiker_dendr,
        accumulators,
        neuron,
        learn,
        **kwargs):
    """ this is the main simulation routine.

    Arguments:

    sim -- a dictionary containing the following simulation parameters
        start:      starting time
        end:        ending time
        dt:         time step
        I_ext:      function for evaluating externally applied current
        pre_spikes: a list of presynaptic spikes
    spiker       -- the somatic spiker
    spiker_dendr -- the dendritic spiker
    accumulators -- a list of accumulators for saving model variables during the simulation
    neuron       -- a dictionary containing the neuron parameters, default_neuron is 
                    used if none specified
    learn        -- a dictionary contianing the learning parameters, default_learn is used 
                    if none specified
    returns:
    a list of accumulators containing simulation results
    """

    use_seed = kwargs.get('seed', 0)
    np.random.seed(use_seed)

    # restrict to positive weights by default
    normalizer = lambda weights: np.where(weights > 0, weights, 0.0)

    # set some default parameters
    p_backprop      = 1.0
    syn_cond_soma = sim.get('syn_cond_soma', {sym: lambda t: 0.0 for sym in ['E', 'I']})
    # ここはデフォルトのphi()を使っている
    dendr_predictor = kwargs.get('dendr_predictor', phi)

    # これはずっと0
    I_ext = sim.get('I_ext', step_current(np.array([[sim['start'], 0.0]])))

    # ensure numpy arrays, for fancy indexing
    pre_spikes = sim['pre_spikes']
    for i, pre_sp in enumerate(pre_spikes):
        pre_spikes[i] = np.array(pre_sp)

    n_syn = len(pre_spikes)
    for key in ['eps', 'eta', 'tau_delta']:
        if not isinstance(learn[key], collections.Iterable):
            learn[key] = np.array([learn[key] for _ in range(n_syn)])
    for acc in accumulators:
        acc.prepare_arrays(n_syn)
        
    t_start = sim['start']
    t_end   = sim['end']
    dt      = sim['dt']
    
    U0 = neuron['E_L']

    curr = {
        't': t_start,
        'y': np.concatenate((np.array([U0, neuron['E_L'], neuron['E_L']]),
                             np.zeros(2 * n_syn)))
        # yは203個になる
        # 最初の3つは[U, V, V_w_star]
        # それ以降は[dV_dws, dV_w_star_dws]の繰り返し
    }
    last_spike = {
        't': float('-inf'),
        'y': curr['y']
    }
    last_spike_dendr = {
        't': float('-inf'),
        'y': curr['y']
    }

    weights = np.array(learn['eps'])

    g_E_Ds         = np.zeros(n_syn)
    syn_pots_sums  = np.zeros(n_syn)
    PIVs           = np.zeros(n_syn)
    deltas         = np.zeros(n_syn)
    weight_updates = np.zeros(n_syn)
    
    while curr['t'] < t_end - dt / 2:
        # currにはtとyが入っている.
        # curr['y'] = (203,)
        
        #print("t={}, t_end={}".format(curr['t'], t_end)) #..
        
        # for each synapse: is there a presynaptic spike at curr['t']?
        # 各ニューロンのpre synapseニューロンが発火したかどうか
        curr_pres = np.array([np.sum(np.isclose(pre_sp, curr['t'],
                                                rtol=1e-10,
                                                atol=1e-10)) for pre_sp in pre_spikes])
        # (100,)の 0or1 (2以上の場合もありうる)

        g_E_Ds = g_E_Ds + curr_pres * weights
        g_E_Ds = g_E_Ds - dt * g_E_Ds / neuron['tau_s'] # 時定数で減衰
        # (100,)

        # 現在時刻tより前のpre synapse発火タイミングでのexponentialを加算する
        syn_pots_sums = np.array(
            [np.sum(np.exp(-(curr['t'] - pre_sp[pre_sp <= curr['t']]) / neuron['tau_s'])) \
             for pre_sp in pre_spikes])
        # (100,)

        # is there a postsynaptic spike at curr['t']?
        if curr['t'] - last_spike['t'] < neuron['tau_ref']:
            # refractory period中だった場合
            does_spike = False
        else:
            does_spike = spiker(curr=curr, dt=dt)
            # sineタスクの場合はdoes_spikeがtrueになることが無い

        if does_spike:
            last_spike = {'t': curr['t'],
                          'y': curr['y']}

        # does the dendrite detect a spike?
        dendr_spike = spiker_dendr(curr=curr,
                                   last_spike=last_spike,
                                   last_spike_dendr=last_spike_dendr)
        # floatの値, phi(U) * dt

        if dendr_spike:
            last_spike_dendr = {'t': curr['t'],
                                'y': curr['y']}

        # dendritic prediction
        # V_w_starから予測 (sigmoidを使ったphi(V_w_star))
        dendr_pred = dendr_predictor(curr['y'][2], neuron)
        # 指定されたh or phi_prime(V_w_star) / phi(V_w_star)
        h = kwargs.get('h', phi_prime(curr['y'][2], neuron) / phi(curr['y'][2], neuron))

        # update weights
        # 重みの更新
        # dV_w_star_dws
        # (delta_factorは1.0)
        pos_PIVs = neuron['delta_factor'] * float(dendr_spike) / dt * h * curr['y'][4::2]
        neg_PIVs = dendr_pred                                       * h * curr['y'][4::2]
        PIVs = pos_PIVs - neg_PIVs # (phi(U) - phi(V_w_star)) * h * dV_w_star_dws
        deltas += dt * (PIVs - deltas) / learn['tau_delta']
        weight_updates = learn['eta'] * deltas
        weights = normalizer(weights + weight_updates)

        # advance state: integrate from curr['t'] to curr['t']+dt
        curr_I = I_ext(curr['t'])
        args = (curr['y'],
                curr['t'],
                curr['t'] - last_spike['t'],
                g_E_Ds,
                syn_pots_sums,
                curr_I,
                neuron,
                syn_cond_soma,
                p_backprop)
        curr['y'] += dt * urb_senn_rhs(*args)
        curr['t'] += dt

        # save state
        vals = {
            'g_E_Ds'        : g_E_Ds,
            'syn_pots_sums' : syn_pots_sums,
            'y'             : curr['y'],
            'spike'         : float(does_spike),
            'dendr_pred'    : dendr_pred,
            'h'             : h,
            'PIVs'          : PIVs,
            'pos_PIVs'      : pos_PIVs,
            'neg_PIVs'      : neg_PIVs,
            'dendr_spike'   : float(dendr_spike),
            'pre_spikes'    : curr_pres,
            'weights'       : weights,
            'weight_updates': weight_updates,
            'deltas'        : deltas,
            'I_ext'         : curr_I
        }
        for acc in accumulators:
            acc.add(curr['t'], **vals)

    for acc in accumulators:
        acc.cleanup()
        acc.add_variable('seed', use_seed)

    return accumulators


def task(p):
    n_syn = p["n_syn"]

    # eta, epsは上がいているのでtau_deltaのみデフォルトを使用
    learn = get_default("learn")
    learn["eps"] = 1e-1 / n_syn
    learn["eta"] = 1e-3 / n_syn

    # alpha, beta, r_max, g_Sを上書き
    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = p["alpha"]
    neuron["phi"]["beta"]  = p["beta"]
    neuron["phi"]["r_max"] = p["r_max"]
    neuron["g_S"]          = p["g_S"]

    epochs    = 4
    l_c       = 6
    eval_c    = 2
    cycles    = epochs * l_c + (epochs + 1) * eval_c # 34
    cycle_dur = p["cycle_dur"] # 100
    t_end     = cycles * cycle_dur # 3400

    def exc_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c:
            return 0.0
        else:
            return ((1 + np.sin(np.pi / 2 + t / t_end * cycles * 2 * np.pi)) \
                    * 2e-3 * 1 + 8e-3) * p["g_factor"]

    def inh_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c:
            return 0.0
        else:
            return 8e-2 * p["g_factor"]

    dt  = 0.05 # ms
    f_r = 0.01 # 10Hz
    t_pts = np.arange(0, t_end / cycles, dt)

    poisson_spikes = [t_pts[np.random.rand(t_pts.shape[0]) < f_r * dt] for _ in range(n_syn)]
    poisson_spikes = [[] if spikes.shape[0] == 0 else np.concatenate(
        [np.arange(spike, t_end, cycle_dur) for spike in spikes]) for spikes in poisson_spikes]
    
    for train in poisson_spikes:
        train.sort()

    my_s = {
        'start'        : 0.0,
        'end'          : t_end,
        'dt'           : dt,
        'pre_spikes'   : poisson_spikes, # 100個の各ニューロンの発火時刻列
        'syn_cond_soma': {'E': exc_soma_cond,
                          'I': inh_soma_cond},
        'I_ext'        : lambda t: 0.0
    }

    seed = 0
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=40)]

    accums = run(my_s,
                 get_fixed_spiker(np.array([])), # spiker
                 get_phi_U_learner(neuron, dt),  # sipker_dendr
                 accs,
                 neuron=neuron,
                 seed=seed,
                 learn=learn)
    
    # TODO: create directrory if not exists
    #..dump((seed, accums), "results/" + p['ident'])


params = {}

params["n_syn"]     = 100
params["g_factor"]  = 100
params["cycle_dur"] = 100

params["g_S"]       = 0.0
params["alpha"]     = -55.0
params["beta"]      = 0.25
params["r_max"]     = 0.35

task(params)
