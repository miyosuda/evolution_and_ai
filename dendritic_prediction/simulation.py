import collections

import numpy as np

from helper import get_default
from model import get_spike_currents, phi, phi_prime, urb_senn_rhs
from util import step_current


def run(sim,
        spiker,
        spiker_dendr,
        accumulators,
        neuron=None,
        learn=None,
        normalizer=None,
        **kwargs):
    """ this is the main simulation routine, can either be called directly, or with
    the convenience routine do in helper.py

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
    voltage_clamp   = kwargs.get('voltage_clamp', False)
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
    
    if voltage_clamp:
        U0 = kwargs['U_clamp']
    else:
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
        print("t={}, t_end={}".format(curr['t'], t_end)) #..
        
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
        neg_PIVs = dendr_pred * h * curr['y'][4::2]
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
                voltage_clamp,
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
