import os
import numpy
import sys

skipthoughts_path = 'theano/skip-thoughts'
sys.path.append(skipthoughts_path)
import skipthoughts

if __name__ == '__main__':

    model = skipthoughts.load_model(dirname='data/raw', fname_umodel='uni_skip.npz', fname_bmodel='bi_skip.npz')
    os.system('mkdir -p data/test')

    X = ['robots']
    features = skipthoughts.encode(model, X, use_norm=False, use_eos=False)
    numpy.save('data/test/features_oneWord_normFalse_eosFalse.npy', features)

    X = ['robots']
    features = skipthoughts.encode(model, X, use_norm=True, use_eos=False)
    numpy.save('data/test/features_oneWord_normTrue_eosFalse.npy', features)

    X = ['robots']
    features = skipthoughts.encode(model, X, use_norm=False, use_eos=True)
    numpy.save('data/test/features_oneWord_normFalse_eosTrue.npy', features)

    X = ['robots']
    features = skipthoughts.encode(model, X, use_norm=True, use_eos=True)
    numpy.save('data/test/features_oneWord_normTrue_eosTrue.npy', features)

    X = ['robots are cool']
    features = skipthoughts.encode(model, X, use_norm=False, use_eos=False)
    numpy.save('data/test/features_normFalse_eosFalse.npy', features)

    X = ['robots are cool']
    features = skipthoughts.encode(model, X, use_norm=True, use_eos=False)
    numpy.save('data/test/features_normTrue_eosFalse.npy', features)

    X = ['robots are cool']
    features = skipthoughts.encode(model, X, use_norm=False, use_eos=True)
    numpy.save('data/test/features_normFalse_eosTrue.npy', features)

    X = ['robots are cool']
    features = skipthoughts.encode(model, X, use_norm=True, use_eos=True)
    numpy.save('data/test/features_normTrue_eosTrue.npy', features)