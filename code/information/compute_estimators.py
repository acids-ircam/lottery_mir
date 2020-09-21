# -*- coding: utf-8 -*-

import torch
import numpy as np
from information.edge import EDGE
from information.estimators_simple import mutual_information_simple
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed

NUM_CORES = 4

mi_estimators = {'regress':mutual_info_regression,
                 'simple':mutual_information_simple, 
                 'edge':EDGE}

def compute_mutual_information(x, y, estimator='edge', parallel=True):
    mi_func = mi_estimators[estimator]
    def select_dim(y_t, dim):
        if (len(y_t.shape) == 2):
            return y_t[:, dim][:, np.newaxis]
        return y_t[:, dim]
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    # Perform parallel operation
    if (parallel):
        mi_vals = np.array(Parallel(n_jobs=NUM_CORES)
                           (delayed(mi_func)(x, select_dim(y, dim))
                            for dim in range(y.shape[1])))
        mi_vals = torch.from_numpy(mi_vals)
    else:
        mi_vals = torch.zeros(y.shape[1])
        for dim in range(y.shape[1]):
            cur_y = select_dim(y, dim)
            mi_vals[dim] = mi_func(x, cur_y)
            print(mi_vals[dim])
    return mi_vals