import os
import numpy
import sys

skipthoughts_path = 'skip-thoughts'
sys.path.append(skipthoughts_path)
import skipthoughts

# You will have to change path_to_models and path_to_tables variables
# in the ./skip-thoughts directory containing the git clone
# of the original skip-thought repositery.
# You will have to change them to `data/skip-thoughts/raw/` or to
# the directory you choosed to download the initial (raw) data.

if __name__ == '__main__':

    model = skipthoughts.load_model()
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