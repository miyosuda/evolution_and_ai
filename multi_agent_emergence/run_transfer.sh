#!/bin/sh

if [ $1 -eq 0 ]; then
    python3 bin/examine.py examples/lock_and_return.jsonnet examples/lock_and_return.npz
elif [ $1 -eq 1 ]; then
    python3 bin/examine.py examples/sequential_lock.jsonnet examples/sequential_lock.npz
elif [ $1 -eq 2 ]; then
    python3 bin/examine.py examples/blueprint.jsonnet examples/blueprint.npz
elif [ $1 -eq 3 ]; then
    python3 bin/examine.py examples/shelter.jsonnet examples/shelter.npz
fi


