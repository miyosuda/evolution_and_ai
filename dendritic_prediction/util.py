import numpy as np
from IPython import embed

from helper import get_default
from model import phi, phi_prime


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
    return lambda curr, dt, **kwargs: spikes.shape[0] > 0 and np.min(np.abs(curr['t'] - spikes)) < dt / 2


def get_phi_spiker(neuron=None):
    """
    returns a somatic spiker that implements an inhomogenuous poisson process,
    i.e. a spike will be elicited with probability dt*firing_rate
    """
    if neuron is None:
        neuron = get_default("neuron")
    return lambda curr, dt, **kwargs: phi(curr['y'][0], neuron) * dt >= np.random.rand()


def get_inst_backprop():
    """
    returns a dendritic spiker that detects a spike whenever there was one
    at the soma, i.e. instantaneous perfect backpropagation
    """
    def inst_backprop(curr, last_spike, **kwargs):
        return np.isclose(curr['t'], last_spike['t'], atol=1e-10, rtol=1e-10)
    return inst_backprop


def get_dendr_spike_det(thresh, tau_ref=10.0):
    """
    returns a dendritic spiker that is based on a simple thresholding of the
    dendritic voltage, together with an associated refractory period
    """
    def dendr_spike_det(curr, last_spike_dendr, **kwargs):
        return curr['y'][1] > thresh and (curr['t'] - last_spike_dendr['t'] > tau_ref)
    return dendr_spike_det


def get_dendr_spike_det_dyn_ref(thresh, tau_ref_0, theta_0):
    """
    returns a dendritic spiker that, similarly to the above one, thresholds the
    dendritic voltage, but the current refractory period is a function of the
    current dendritic voltage, computed by a decaying exponential, i.e. high
    voltages will lead to a faster sampling than shallow threshold crossings
    """
    def dendr_spike_det_dyn_ref(curr, last_spike_dendr, **kwargs):
        if curr['y'][1] > thresh:
            curr_ref = tau_ref_0 * np.exp(-(curr['y'][1] - thresh) / theta_0)
            return curr['t'] - last_spike_dendr['t'] > curr_ref
        else:
            return False
    return dendr_spike_det_dyn_ref


def get_phi_U_learner(neuron, dt, factor=1.0):
    """
    returns a dendritic learning signal that has full access to the somatic voltage
        and rate-function phi
    i.e. the average version of inst_backprop if somatic spikes are initiated
        based on an inhomogenuous poisson process with rate phi
    """
    def phi_U_learner(curr, **kwargs):
        return factor * phi(curr['y'][0], neuron) * dt
    return phi_U_learner


def step_current(steps):
    """
    defines a step current, i.e. an externally applied current defined by step
    changes, e.g. [][0,0],[2,4],[5,-1]] defines a current that is initially zero,
    stepped to 4 at 2ms, and stepped to -1 at 5ms
    """
    return lambda t: steps[steps[:, 0] <= t, 1][-1]


def get_periodic_current(first, interval, width, dc_on, dc_off=0.0):
    """defines a periodic current, defined by an on voltage dc_on and an off voltage
    dc_off. on periods correspond to symmetric intervals with a certain width around time points
    first, first+interval, first+2*interval, ...
    """
    def I_ext(t):
        if t >= (first - width / 2) and np.mod(t - first + width / 2, interval) < width:
            return dc_on
        else:
            return dc_off
    return I_ext
