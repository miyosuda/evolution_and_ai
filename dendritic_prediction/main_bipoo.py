# -*- coding: utf-8 -*-
"""
Here we reproduce experiments reported in
"Synaptic Modifications in Cultured Hippocampal Neurons:
Dependence on Spike Timing, Synaptic Strength, and
Postsynaptic Cell Type"
Guo-qiang Bi and Mu-ming Poo
The Journal of Neuroscience, 1998

Specifically, we investigate the basic spike-timing dependence of plasticity
by manipulating the relative difference of pre- and postsynaptic spikes
(Figure 7). The data from this figure can be found in
the "data" folder.

Approximate runtime on an Intel Xeon X3470 machine (4 CPUs, 8 threads):
< 2min

Running this file should produce 101 .p files.

Afterwards, code in the corresponding
IPython notebook will produce a figure showing experimental data and
simulation results next to each other.
"""

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from collections import OrderedDict
from simulation import run


def fit(inputs):
    repetition_i = inputs[0]
    p            = inputs[1]
    
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
        'pre_spikes': [spikes + p["delta"]],
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
                     get_dendr_spike_det(-50.0),
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

    dump((accums, values), "results/" + p['ident'])


params = OrderedDict()
params["h1"] = [False, True]
params["delta"] = np.linspace(-100.0, 100.0, 101)

file_prefix = 'bi_poo_fit'

do(fit,
   params,
   file_prefix,
   withmp=True)
