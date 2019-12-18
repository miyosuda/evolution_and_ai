from functools import reduce
import numpy as np


def dump(res, ident):
    import _pickle as cPickle
    cPickle.dump(res, open('{0}.p'.format(ident), 'wb'))


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
        self.keys = keys
        self.init_size = init_size
        self.i = interval
        self.j = 0
        self.size = init_size
        self.interval = interval
        self.t = np.zeros(init_size, np.float32)
        self.y_keep = y_keep

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


def do(func, params, file_prefix, create_notebooks=True, **kwargs):
    from parallelization import run_tasks

    runs, base_str = construct_params(params, file_prefix)
    
    run_tasks(params=runs, runTask=func, **kwargs)


def construct_params(params, prefix=''):
    from itertools import product
    from operator import add

    ids = tuple(params.keys())
    values = tuple(params.values())

    if prefix.endswith("_"):
        prefix = prefix[:-1]

    base_str = prefix + reduce(add, ['_{0}_{{{1}}}'.format(ids[i], i) for i in range(len(ids))], "")

    combinations = product(*values)
    concat_params = []
    for comb in combinations:
        curr = {id: val for (id, val) in zip(ids, comb)}
        curr['ident'] = base_str.format(*comb)
        concat_params.append(curr)

    return concat_params, base_str
