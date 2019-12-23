# -*- coding: utf-8 -*-
import numpy as np
import collections

from model import phi, phi_prime, urb_senn_rhs


class BooleanAccumulator:
    def __init__(self, keys):
        self.keys = keys
        self.res = { key: np.array([]) for key in keys }

    def add(self, curr_t, **vals):
        for key in self.keys:
            if vals[key]:
                self.res[key] = np.append(self.res[key], curr_t)

    def prepare_arrays(self, n_syn):
        pass

    def cleanup(self):
        pass

    def add_variable(self, name, val):
        self.res[name] = val


class PeriodicAccumulator:
    def _get_size(self, key):
        if key == 'y':
            if self.y_keep is not None:
                return self.y_keep
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

    def __init__(self, keys, interval=1, init_size=1024, y_keep=None):
        self.keys      = keys
        self.init_size = init_size
        self.i         = interval
        self.j         = 0
        self.size      = init_size
        self.interval  = interval
        self.t         = np.zeros(init_size, np.float32)
        self.y_keep    = y_keep

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
                    self.res[key] = np.vstack(
                        (self.res[key], np.zeros(self.res[key].shape, np.float32)))
                self.size = self.size * 2

            for key in self.keys:
                if key == 'y' and self.y_keep is not None:
                    self.res[key][self.j, :] = np.atleast_2d(vals[key][:self.y_keep])
                else:
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


#def get_all_save_keys():
#    """ all variables that can be recorded during a simulation """
#    return ['g_E_Ds',
#            'syn_pots_sums',
#            'y',
#            'spike',
#            'dendr_pred',
#            'h',
#            'PIVs',
#            'pos_PIVs',
#            'neg_PIVs',
#            'dendr_spike',
#            'pre_spikes',
#            'weights',
#            'weight_updates',
#            'deltas',
#            'I_ext']


def get_fixed_spiker(spikes):
    """
    returns a somatic spiker that implements a fixed spiker, i.e. it will return true
    whenever the current time is listed in the argument spikes
    """
    return lambda curr, dt, **kwargs: spikes.shape[0] > 0 and \
        np.min(np.abs(curr['t'] - spikes)) < dt / 2


#def get_phi_U_learner(neuron, dt, factor=1.0):
#    """
#    returns a dendritic learning signal that has full access to the somatic voltage
#        and rate-function phi
#    i.e. the average version of inst_backprop if somatic spikes are initiated
#        based on an inhomogenuous poisson process with rate phi
#    """
#    def phi_U_learner(curr, **kwargs):
#        return factor * phi(curr['y'][0], neuron) * dt
#    return phi_U_learner


def get_dendr_spike_det(thresh, tau_ref=10.0):
    """
    returns a dendritic spiker that is based on a simple thresholding of the
    dendritic voltage, together with an associated refractory period
    """
    def dendr_spike_det(curr, last_spike_dendr, **kwargs):
        # Vがスレッショルドを超えたかつ最後のdendr spikeからの間隔がtau_ref以上なら
        return curr['y'][1] > thresh and (curr['t'] - last_spike_dendr['t'] > tau_ref)
    return dendr_spike_det


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
        neuron=None,
        learn=None,
        normalizer=None,
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
    normalizer   -- a function to normalize synaptic weights, e.g. the default normalizer
                    ensures non-negative weights
    returns:
    a list of accumulators containing simulation results
    """

    use_seed = kwargs.get('seed', 0)
    np.random.seed(use_seed)

    if neuron is None:
        neuron = get_default('neuron')

    if learn is None:
        learn = get_default('learn')

    # restrict to positive weights by default
    if normalizer is None:
        normalizer = lambda weights: np.where(weights > 0, weights, 0.0)

    # set some default parameters
    p_backprop      = kwargs.get('p_backprop', 1.0)
    syn_cond_soma = sim.get('syn_cond_soma', {sym: lambda t: 0.0 for sym in ['E', 'I']})
    dendr_predictor = kwargs.get('dendr_predictor', phi)

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
                             np.zeros(2 * n_syn)))}
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
        #print("t={}, t_end={}".format(curr['t'], t_end)) #..
        
        # for each synapse: is there a presynaptic spike at curr['t']?
        curr_pres = np.array([np.sum(np.isclose(pre_sp, curr['t'],
                                                rtol=1e-10,
                                                atol=1e-10)) for pre_sp in pre_spikes])

        g_E_Ds = g_E_Ds + curr_pres * weights
        g_E_Ds = g_E_Ds - dt * g_E_Ds / neuron['tau_s']

        syn_pots_sums = np.array(
            [np.sum(np.exp(-(curr['t'] - pre_sp[pre_sp <= curr['t']]) / neuron['tau_s'])) \
             for pre_sp in pre_spikes])

        # is there a postsynaptic spike at curr['t']?
        if curr['t'] - last_spike['t'] < neuron['tau_ref']:
            does_spike = False
        else:
            does_spike = spiker(curr=curr, dt=dt)

        if does_spike:
            last_spike = {'t': curr['t'],
                          'y': curr['y']}

        # does the dendrite detect a spike?
        dendr_spike = spiker_dendr(curr=curr,
                                   last_spike=last_spike,
                                   last_spike_dendr=last_spike_dendr)

        if dendr_spike:
            last_spike_dendr = {'t': curr['t'], 'y': curr['y']}

        # dendritic prediction
        dendr_pred = dendr_predictor(curr['y'][2], neuron)
        h = kwargs.get('h', phi_prime(curr['y'][2], neuron) / phi(curr['y'][2], neuron))

        # update weights
        pos_PIVs = neuron['delta_factor'] * float(dendr_spike) / dt * h * curr['y'][4::2]
        neg_PIVs = dendr_pred                                       * h * curr['y'][4::2]
        PIVs = pos_PIVs - neg_PIVs
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


def fit(p):
    values = {
        True: {
            "alpha" : -55.0,
            "beta"  : 0.4,
            "r_max" : 0.3
        },
        False: {
            "alpha" : -59.0,
            "beta"  : 0.5,
            "r_max" : 0.17
        }
    }

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = values[p["h1"]]["r_max"]
    neuron["phi"]['alpha'] = values[p["h1"]]["alpha"]
    neuron["phi"]['beta']  = values[p["h1"]]["beta"]
    
    learn = get_default("learn")
    if not p["h1"]:
        learn["eta"] = learn["eta"] * 2.5
    else:
        learn["eta"] = learn["eta"] * 1.3

    spikes = np.array([101.0])

    my_s = {
        'start'     : 0.0,
        'end'       : 300.0,
        'dt'        : 0.05,
        'pre_spikes': [spikes + p["delta"]], # preの発火タイミング
        'I_ext'     : lambda t: 0.0
    }

    seed = 1
    accs = [
        PeriodicAccumulator(['y', 'weights'], interval=10),
        BooleanAccumulator(['spike', 'dendr_spike', 'pre_spikes'])
    ]
    if p["h1"]:
        # h=1.0を指定する場合
        accums = run(my_s,
                     get_fixed_spiker(spikes),
                     get_dendr_spike_det(-50.0), # ref期間外かつVが-50.0を超えたらspike
                     accs,
                     seed=seed,
                     neuron=neuron,
                     learn=learn,
                     h=1.0)
    else:
        # 
        accums = run(my_s,
                     get_fixed_spiker(spikes),
                     get_dendr_spike_det(-50.0),
                     accs,
                     seed=seed,
                     neuron=neuron,
                     learn=learn)

    #..dump((accums, values), "results/" + p['ident'])


deltas = np.linspace(-100.0, 100.0, 101) # -100ms ~ 100ms

for delta in deltas:
    params = {}
    params["h1"] = False
    params["delta"] = delta # ms
    
    fit(params)
    print("delta={}".format(delta))
