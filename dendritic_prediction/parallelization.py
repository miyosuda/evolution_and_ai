import itertools as it
import multiprocessing as mp


def run_tasks(params, runTask, repetitions=1, withmp=True):
    '''
    params      - a list of all param values to be simulated
    repetitions - integer number repetitions per param
    withmp      - boolean enables multiprocessing, otherwise run serially
    '''
    # below is a generator expression that returns tuples of parameters to be passed
    # into the run_task function. It will have repetitions*len(params) elements.
    all_reps_params = ((rep_i, param) for param, rep_i in it.product(params, range(repetitions)))

    print("running {0} simulations".format(repetitions * len(params)))
    
    if withmp:
        pool = mp.Pool(mp.cpu_count())
        pool.map(runTask, all_reps_params)
    else:
        for pair in all_reps_params:
            runTask(pair)
