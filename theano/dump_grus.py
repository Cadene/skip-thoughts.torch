import os
import argparse
import numpy as np
import cPickle as pkl

import theano
import theano.tensor as tensor

import sys
skipthoughts_path = 'theano/skip-thoughts'
sys.path.append(skipthoughts_path)
import skipthoughts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='data')
    args = parser.parse_args()

    ############################################
    # Directories and Paths to files
    ############################################

    dir_raw = os.path.join(args.dirname, 'raw')
    path_umodel = os.path.join(dir_raw, 'uni_skip.npz')
    path_bmodel = os.path.join(dir_raw, 'bi_skip.npz')
    path_umodel_opt = os.path.join(dir_raw, 'uni_skip.npz.pkl')
    path_bmodel_opt = os.path.join(dir_raw, 'bi_skip.npz.pkl')
    
    dir_interim = os.path.join(args.dirname, 'interim')
    ufilename = 'uparams_%s.npy'
    bfilename = 'bparams_%s.npy'
    os.system('mkdir -p ' + dir_interim)

    ############################################
    # Format params of uni-skip model
    ############################################

    with open(path_umodel_opt, 'rb') as f:
        uoptions = pkl.load(f)

    print('Load skip-thoughts with Theano')
    uparams = skipthoughts.init_params(uoptions)
    uparams = skipthoughts.load_params(path_umodel, uparams)

    for k in uparams.keys():
        path_params = os.path.join(dir_interim, ufilename % k)
        print('Save ' + ufilename % k)
        np.save(path_params, uparams[k]) 

    ############################################
    # Format params of bi-skip model
    ############################################

    with open(path_bmodel_opt, 'rb') as f:
        boptions = pkl.load(f)

    print('Load skip-thoughts with Theano')
    bparams = skipthoughts.init_params_bi(boptions)
    bparams = skipthoughts.load_params(path_bmodel, bparams)
    print(bparams.keys())

    for k in bparams.keys():
        path_params = os.path.join(dir_interim, bfilename % k)
        print('Save ' + bfilename % k)
        np.save(path_params, bparams[k]) 
