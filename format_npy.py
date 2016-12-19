import os
import numpy as np
import argparse

def format_npy(utable):
    nb_words = utable.shape[0]
    nb_embedding = utable[0].shape[1]
    utable_npy = np.empty([nb_words, nb_embedding])
    for i in xrange(nb_words):
        if utable[i].shape[0] == nb_embedding:
            utable_npy[i] = utable[i]
        elif utable[i].shape[1] == nb_embedding:
            utable_npy[i] = utable[i][0]
        else:
            print 'Warning' + utable[i].shape[1]
    return utable_npy

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='data/raw')
    args = parser.parse_args()

    dir_raw = os.path.join(args.dirname, 'raw')
    path_utable = os.path.join(dir_raw, 'utable.npy')
    path_btable = os.path.join(dir_raw, 'btable.npy')

    dir_interim = os.path.join(args.dirname, 'interim')
    path_utable_npy = os.path.join(dir_interim, 'utable.npy')
    path_btable_npy = os.path.join(dir_interim, 'btable.npy')
    os.system('mkdir -p ' + dir_interim)

    utable = np.load(path_utable)
    utable_npy = format_npy(utable)    
    np.save(path_utable_npy, utable_npy)

    btable = np.load(path_btable)
    btable_npy = format_npy(btable) 
    np.save(path_btable_npy, btable_npy)
