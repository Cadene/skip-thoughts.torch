import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import numpy
from collections import OrderedDict

urls = {}
urls['dictionary'] = 'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt'
urls['utable']     = 'http://www.cs.toronto.edu/~rkiros/models/utable.npy'
urls['uni_skip']   = 'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz'

class UniSkip(nn.Module):

    def __init__(self, dir_st, vocab, process_lengths=False, dropout=0):
        super(UniSkip, self).__init__()
        self.dir_st  = dir_st
        self.vocab   = vocab
        self.process_lengths = process_lengths
        self.dropout = dropout
        # Modules
        self.embedding = self._load_embedding() 
        self.gru       = self._load_gru()
        # Remove bias_ih_l0 (== zero all the time)
        del self.gru._parameters['bias_ih_l0']
        del self.gru._all_weights[0][2]
 
    def _load_dictionary(self):
        path_dico = os.path.join(self.dir_st, 'dictionary.txt')
        if not os.path.exists(path_dico):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls['dictionary'], self.dir_st))
        with open(path_dico, 'r') as handle:
            dico_list = handle.readlines()
        dico = {word.strip():idx for idx,word in enumerate(dico_list)}
        return dico
 
    def _load_emb_params(self):
        path_params = os.path.join(self.dir_st, 'utable.npy')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls['utable'], self.dir_st))
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
 
    def _load_gru_params(self):
        path_params = os.path.join(self.dir_st, 'uni_skip.npz')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls['uni_skip'], self.dir_st))
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
 
    def _make_emb_state_dict(self, dictionary, parameters):
        weight = torch.zeros(len(self.vocab)+1, 620) # first dim = zeros -> +1
        unknown_params = parameters[dictionary['UNK']]
        for id_weight, word in enumerate(self.vocab):
            if word in dictionary:
                id_params = dictionary[word]
                params = parameters[id_params]
            else:
                print('Warning: word `{}` not in dictionary'.format(word))
                params = unknown_params
            weight[id_weight+1] = torch.from_numpy(params)
        state_dict = OrderedDict({'weight':weight})
        return state_dict
 
    def _make_gru_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0']   = torch.zeros(7200) # must stay equal to 0
        s['bias_hh_l0']   = torch.zeros(7200) 
        s['weight_ih_l0'] = torch.zeros(7200, 620)
        s['weight_hh_l0'] = torch.zeros(7200, 2400)
        s['weight_ih_l0'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_hh_l0'][:4800]   = torch.from_numpy(p['encoder_b'])
        s['bias_hh_l0'][4800:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][4800:] = torch.from_numpy(p['encoder_Ux']).t()             
        return s
 
    def _load_embedding(self):
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                      embedding_dim=620,
                                      padding_idx=0) # -> first_dim = zeros
        dictionary = self._load_dictionary()
        parameters = self._load_emb_params()
        state_dict = self._make_emb_state_dict(dictionary, parameters)
        self.embedding.load_state_dict(state_dict)
        return self.embedding
 
    def _load_gru(self):
        self.gru = nn.GRU(input_size=620,
                          hidden_size=2400,
                          batch_first=True,
                          dropout=self.dropout)
        parameters = self._load_gru_params()
        state_dict = self._make_gru_state_dict(parameters)
        self.gru.load_state_dict(state_dict)
        return self.gru
 
    def _select_last(self, input, lengths):
        x = Variable(input.data.new().resize_((input.size(0), input.size(2))))
        for i in range(input.size(0)):
            x[i] = input[i,lengths[i]-1]
        return x
 
    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
        return lengths

    def forward(self, input, lengths=None):
        if lengths is None or self.process_lengths:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        x, hn = self.gru(x) # seq2seq
        if lengths:
            x = self._select_last(x, lengths)
        return x
 
if __name__ == '__main__':
    dir_st = '/local/cadene/data/skip-thoughts'
    vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
    uniskip = UniSkip(dir_st, vocab)
 
    # batch_size x seq_len
    input = Variable(torch.LongTensor([
        [1,2,3,4,5,0,0],
        [6,1,2,3,3,4,5],
        [6,1,2,3,3,4,5]
    ]))
    uniskip.eval()

    # batch_size x 2400
    output_seq2vec = uniskip(input, lengths=[5,7,7])

    # batch_size x 2400
    uniskip.process_lengths = True
    output_seq2vec2 = uniskip(input)

    # batch_size x seq_len x 2400
    output_seq2seq = uniskip(input)