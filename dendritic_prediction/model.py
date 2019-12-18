import numpy as np


def get_spike_currents(U, t_post_spike, n):
    """
    this function implements our simplified action potential currents (Eq. 3)
    if t_post_spike is the time since the last spike
        0 <= t_post_spike < t_rise: Na+ conductance
        t_rise <= t_post_spike < t_fall: K+ conductance
    parameters:
    U -- the current somatic potential
    t_post_spike -- time since last spike
    n -- the dictionary containing neuron parameters, e.g. contents of default_neuron.json
    returns:
    action potential current
    """
    current = 0.0
    if 0.0 <= t_post_spike < n['t_rise']:
        current += -n['g_Na'] * (U - n['E_Na'])
    if n['t_rise'] <= t_post_spike < n['t_fall']:
        current += -n['g_K'] * (U - n['E_K'])
    return current


def phi(U, n):
    """
    the transfer function somatic voltage -> firing rate which determines the
    firing intensity of the inhomogenuous poisson process, in our case a sigmoidal function (Eq. 2)
    parameters:
    U -- the current somatic potential
    n -- the dictionary containing neuron parameters, e.g. contents of default_n.json
    returns:
    firing rate
    """
    phi_params = n['phi']
    return phi_params['r_max'] / (1 + np.exp(-phi_params['beta'] * (U - phi_params['alpha'])))


def phi_prime(U, n):
    """
    computes the first derivative of the sigmoidal firing rate function
    """
    phi_params = n['phi']
    num = np.exp((U + phi_params["alpha"]) * phi_params["beta"]) * \
        phi_params["r_max"] * phi_params["beta"]
    denom = (np.exp(U * phi_params["beta"]) + np.exp(phi_params["alpha"] * phi_params["beta"]))**2
    return num / denom


def urb_senn_rhs(y, t, t_post_spike, g_E_Ds, syn_pots_sums, I_ext, n, g_syn_soma, voltage_clamp, p_backprop):
    """
    computes the right hand side describing how the system of differential equations
    evolves in time, used for Euler integration
    parameters:
    y -- the current state (see first line of code)
    t -- the current time
    t_post_spike -- time since last spike
    g_E_D -- current excitatory conductance from dendritic synapses
    syn_pots_sum -- current value of input spike train convolved with exponential decay
        see Eq. 5 and text thereafter
    I_ext -- current externally applied current (to the soma)
    n -- the dictionary containing neuron parameters
    g_syn_soma -- two functions inside a dict returning time-dependent somatic conductances
        coming from synaptic input (with exc. and inh. reversal potentials)
    voltage_clamp -- a boolean indicating whether we clamp the somatic voltage or let it evolve
    p_backprop -- a probability p where we set the conductance soma -> dendrite to zero
        with probability (1-p)
    returns:
    the r.h.s. of the system of differential equations
    """
    (U, V, V_w_star) = tuple(y[:3])
    dV_dws, dV_w_star_dws = y[3::2], y[4::2]
    dy = np.zeros(y.shape)

    # U derivative
    if voltage_clamp:
        dy[0] = 0.0
    else:
        syn_input = -g_syn_soma['E'](t) * (U - n['E_E']) - g_syn_soma['I'](t) * (U - n['E_I'])
        dy[0] = -n['g_L'] * (U - n['E_L']) - n['g_D'] * (U - V) + syn_input + I_ext
        if t_post_spike <= n['t_fall']:
            dy[0] = dy[0] + get_spike_currents(U, t_post_spike, n)

    # V derivative
    dy[1] = -n['g_L'] * (V - n['E_L']) - np.sum(g_E_Ds) * (V - n['E_E'])
    if np.random.rand() <= p_backprop:
        dy[1] += -n['g_S'] * (V - U)

    # V_w_star derivative
    dy[2] = -n['g_L'] * (V_w_star - n['E_L']) - n['g_D'] * (V_w_star - V)

    # partial derivatives w.r.t the synaptic weights
    dy[3::2] = -(n['g_L'] + n['g_S'] + g_E_Ds) * dV_dws + \
        n['g_S'] * dV_w_star_dws + (n['E_E'] - V) * syn_pots_sums
    dy[4::2] = -(n['g_L'] + n['g_D']) * dV_w_star_dws + n['g_D'] * dV_dws

    return dy
