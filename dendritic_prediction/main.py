import time
from collections import OrderedDict

import numpy as np

from helper import PeriodicAccumulator, do, dump, get_default
from simulation import run
from util import get_all_save_keys, get_fixed_spiker, get_inst_backprop, \
    get_periodic_current, get_phi_spiker, get_phi_U_learner


def task(inputs):
    repetition_i = inputs[0]
    p = inputs[1]
         
    n_syn = p["n_syn"]

    learn = get_default("learn")
    learn["eps"] = 1e-1 / n_syn
    learn["eta"] = 1e-3 / n_syn

    neuron = get_default("neuron")
    
    neuron["phi"]["alpha"] = p["alpha"]
    neuron["phi"]["beta"]  = p["beta"]
    neuron["phi"]["r_max"] = p["r_max"]
    neuron["g_S"]          = p["g_S"]

    epochs    = 4
    l_c       = 6
    eval_c    = 2
    cycles    = epochs * l_c + (epochs + 1) * eval_c
    cycle_dur = p["cycle_dur"]
    t_end     = cycles * cycle_dur

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

    dt  = 0.05
    f_r = 0.01  # 10Hz
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
        'pre_spikes'   : poisson_spikes,
        'syn_cond_soma': {'E': exc_soma_cond,
                          'I': inh_soma_cond},
        'I_ext'        : lambda t: 0.0
    }

    seed = int(int(time.time() * 1e8) % 1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=40)]

    accums = run(my_s,
                 get_fixed_spiker(np.array([])),
                 get_phi_U_learner(neuron, dt),
                 accs,
                 neuron=neuron,
                 seed=seed,
                 learn=learn)
    # TODO: create directrory if not exists
    dump((seed, accums), "results/" + p['ident'])

params = OrderedDict()
params["n_syn"]     = [50, 100]
params["g_factor"]  = [10]
params["cycle_dur"] = [100]
params["g_S"]       = [0.0, 0.5]
params["alpha"]     = [-50.0, -55.0]
params["beta"]      = [0.2, 0.25]
params["r_max"]     = [0.25, 0.35]

file_prefix = 'sine_task'

do(func=task,
   params=params,
   file_prefix=file_prefix,
   create_notebooks=False,
   withmp=False)
