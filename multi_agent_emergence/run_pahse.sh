#!/bin/bash

phase_data="a_chasing.npz"
if [ $1 -eq 0 ]; then
    phase_data="a_chasing.npz"
elif [ $1 -eq 1 ]; then
    phase_data="b_forts.npz"
elif [ $1 -eq 2 ]; then
    phase_data="c_ramps.npz"
elif [ $1 -eq 3 ]; then
    phase_data="d_ramp_defense.npz"
elif [ $1 -eq 4 ]; then
    phase_data="e_box_surfing.npz"
fi

python3 bin/examine.py examples/hide_and_seek_full.jsonnet examples/hide_and_seek_policy_phases/$phase_data
