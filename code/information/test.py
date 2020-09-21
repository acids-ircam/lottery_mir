# -*- coding: utf-8 -*-

import time
import numpy as np
from estimators_simple import entropy_gaussian
# Import different mutual information estimators
from estimators_simple import mutual_information_simple, mutual_information_sklearn
from estimators_adaptive import mutual_information_kde, mutual_information_sampling
from estimators_eet import mutual_information_npeet
from sklearn.feature_selection import mutual_info_regression
from edge import EDGE
from joblib import Parallel, delayed

NUM_CORES = 8

mi_estimators = [mutual_info_regression,
                 mutual_information_simple, 
                 #mutual_information_npeet,
                 EDGE]
                 #mutual_information_sklearn,
                 #mutual_information_sampling,

def test_mutual_information(function):
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 1000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    start_time = time.time()
    MI_est = function(X, Y)
    end_time = time.time() - start_time
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print('Complete vectors tests')
    print((MI_est, MI_th))
    print(end_time)
    #np.testing.assert_array_less(MI_est, MI_th)
    #np.testing.assert_array_less(MI_th, MI_est  + .3)
    print(Y.shape)
    print('Per-dimension test')
    for dim in range(1):
        cur_Y = Y[:, dim][:, np.newaxis]
        start_time = time.time()
        MI_est = function(X, cur_Y)
        end_time = time.time() - start_time
        print(end_time)
    # Same test for extremely large variables (just test)
    X = np.random.randn(n, 100)
    Y = np.random.randn(n, 1024)
    start_time = time.time()
    #MI_est = function(X, Y)
    end_time = time.time() - start_time
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print('Sub-vectors tests')
    print(end_time)
    #np.testing.assert_array_less(MI_est, MI_th)
    #np.testing.assert_array_less(MI_th, MI_est  + .3)
    print('Per-dimension test')
    for dim in range(10):
        cur_Y = Y[:, dim][:, np.newaxis]
        print(cur_Y.shape)
        start_time = time.time()
        MI_est = function(X, cur_Y)
        end_time = time.time() - start_time
        print(end_time)
        print(MI_est.shape)
    print('Parallel vector tests')
    start_time = time.time()
    params = np.array(Parallel(n_jobs=NUM_CORES)
            (delayed(function)
            (X, Y[:, dim][:, np.newaxis])
            for dim in range(10)))
    end_time = time.time() - start_time
    print(end_time)
    print(params)
    
        

if __name__ == '__main__':
    # Run our tests
    for f in mi_estimators:
        print(f)
        test_mutual_information(f)