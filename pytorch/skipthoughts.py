import os
import sys
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

from gru import BayesianGRU, GRU
from sequential_dropout import SequentialDropout

urls = {}
urls['dictionary'] = 'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt'
urls['utable']     = 'http://www.cs.toronto.edu/~rkiros/models/utable.npy'
urls['uni_skip']   = 'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz'


# class AbstractSkipthoughts(nn.Module):

#     def __init__(self, dir_st, vocab):
#         super(AbstractSkipthoughts, self).__init__()
#         self.dir_st = dir_st
#         self.vocab = vocab
#         # Modules
#         import ipdb; ipdb.set_trace()
        


class AbstractUniSkip(nn.Module):

    def __init__(self, dir_st, vocab, save=True, dropout=0):
        super(AbstractUniSkip, self).__init__()
        self.dir_st = dir_st
        self.vocab = vocab
        self.save = save
        self.dropout = dropout
        self.embedding = self._load_embedding()
        self.rnn = self._load_rnn()
 
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
 
    def _load_rnn_params(self):
        path_params = os.path.join(self.dir_st, 'uni_skip.npz')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls['uni_skip'], self.dir_st))
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
 
    def _load_embedding(self):
        if self.save:
            import hashlib
            import pickle
            # http://stackoverflow.com/questions/20416468/fastest-way-to-get-a-hash-from-a-list-in-python
            hash_id = hashlib.sha256(pickle.dumps(self.vocab, -1)).hexdigest()
            path = '/tmp/uniskip_embedding_'+str(hash_id)+'.pth'
        if self.save and os.path.exists(path):
            self.embedding = torch.load(path)
        else:
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                          embedding_dim=620,
                                          padding_idx=0) # -> first_dim = zeros
            dictionary = self._load_dictionary()
            parameters = self._load_emb_params()
            state_dict = self._make_emb_state_dict(dictionary, parameters)
            self.embedding.load_state_dict(state_dict)
            if self.save:
                torch.save(self.embedding, path)
        return self.embedding

    def _make_emb_state_dict(self, dictionary, parameters):
        weight = torch.zeros(len(self.vocab)+1, 620) # first dim = zeros -> +1
        unknown_params = parameters[dictionary['UNK']]
        nb_unknown = 0
        for id_weight, word in enumerate(self.vocab):
            if word in dictionary:
                id_params = dictionary[word]
                params = parameters[id_params]
            else:
                #print('Warning: word `{}` not in dictionary'.format(word))
                params = unknown_params
                nb_unknown += 1
            weight[id_weight+1] = torch.from_numpy(params)
        state_dict = OrderedDict({'weight':weight})
        if nb_unknown > 0:
            print('Warning: {}/{} words are not in dictionary, thus set UNK'
                  .format(nb_unknown, len(dictionary)))
        return state_dict
 
    def _select_last(self, input, lengths):
        batch_size = input.size(0)
        x = []
        for i in range(batch_size):
            x.append(input[i,lengths[i]-1].view(1, 2400))
        output = torch.cat(x, 0)
        return output
 
    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
        return lengths

    def _load_rnn(self):
        raise NotImplementedError

    def _make_rnn_state_dict(self, p):
        raise NotImplementedError

    def forward(self, input, lengths=None):
        raise NotImplementedError
    

class UniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=True, dropout=0.25):
        super(UniSkip, self).__init__(dir_st, vocab, save, dropout)
        # Remove bias_ih_l0 (== zero all the time)
        # del self.gru._parameters['bias_hh_l0']
        # del self.gru._all_weights[0][3]

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620,
                          hidden_size=2400,
                          batch_first=True,
                          dropout=self.dropout)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0']   = torch.zeros(7200) 
        s['bias_hh_l0']   = torch.zeros(7200) # must stay equal to 0
        s['weight_ih_l0'] = torch.zeros(7200, 620)
        s['weight_hh_l0'] = torch.zeros(7200, 2400)
        s['weight_ih_l0'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:4800]   = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][4800:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][4800:] = torch.from_numpy(p['encoder_Ux']).t()             
        return s

    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        x, hn = self.rnn(x) # seq2seq
        if lengths:
            x = self._select_last(x, lengths)
        return x


class DropUniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=True, dropout=0.25):
        super(DropUniSkip, self).__init__(dir_st, vocab, save, dropout)
        # Modules
        self.seq_drop_x = SequentialDropout(p=self.dropout)
        self.seq_drop_h = SequentialDropout(p=self.dropout)
 
    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih']   = torch.zeros(7200) 
        s['bias_hh']   = torch.zeros(7200) # must stay equal to 0
        s['weight_ih'] = torch.zeros(7200, 620)
        s['weight_hh'] = torch.zeros(7200, 2400)
        s['weight_ih'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih'][:4800]   = torch.from_numpy(p['encoder_b'])
        s['bias_ih'][4800:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh'][4800:] = torch.from_numpy(p['encoder_Ux']).t()             
        return s
 
    def _load_rnn(self):
        self.rnn = nn.GRUCell(620, 2400)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn
 
    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        seq_length = input.size(1) 

        if lengths is None:
            lengths = self._process_lengths(input)

        x = self.embedding(input)

        hx = Variable(x.data.new().resize_((batch_size, 2400)).fill_(0))
        output = []

        for i in range(seq_length):

            if self.dropout > 0:
                input_gru_cell = self.seq_drop_x(x[:,i,:])
                hx = self.seq_drop_h(hx)
            else:
                input_gru_cell = x[:,i,:]

            hx = self.rnn(input_gru_cell, hx)
            output.append(hx.view(batch_size, 1, 2400))

        output = torch.cat(output, 1) # seq2seq

        if lengths:
            output = self._select_last(output, lengths)

        return output


class BayesianUniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=True, dropout=0.25):
        super(BayesianUniSkip, self).__init__(dir_st, vocab, save, dropout)

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['gru_cell.weight_ir.weight'] = torch.from_numpy(p['encoder_W']).t()[:2400]
        s['gru_cell.weight_ii.weight'] = torch.from_numpy(p['encoder_W']).t()[2400:]
        s['gru_cell.weight_in.weight'] = torch.from_numpy(p['encoder_Wx']).t()

        s['gru_cell.weight_ir.bias'] = torch.from_numpy(p['encoder_b'])[:2400]
        s['gru_cell.weight_ii.bias'] = torch.from_numpy(p['encoder_b'])[2400:]
        s['gru_cell.weight_in.bias'] = torch.from_numpy(p['encoder_bx'])

        s['gru_cell.weight_hr.weight'] = torch.from_numpy(p['encoder_U']).t()[:2400]
        s['gru_cell.weight_hi.weight'] = torch.from_numpy(p['encoder_U']).t()[2400:]       
        s['gru_cell.weight_hn.weight'] = torch.from_numpy(p['encoder_Ux']).t()             
        return s
 
    def _load_rnn(self):
        self.rnn = BayesianGRU(620, 2400, dropout=self.dropout)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn
 
    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        x, hn = self.rnn(x) # seq2seq
        if lengths:
            x = self._select_last(x, lengths)
        return x


 
if __name__ == '__main__':
    dir_st = '/local/cadene/data/skip-thoughts'
    vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
    uniskip = BayesianUniSkip(dir_st, vocab)
 
    # batch_size x seq_len
    input = Variable(torch.LongTensor([
        [1,2,3,4,5,0,0],
        [6,1,2,3,3,4,5],
        [6,1,2,3,3,4,5]
    ]))
    print(input.size())
    uniskip.eval()

    # batch_size x 2400
    output_seq2vec = uniskip(input, lengths=[5,7,7])
    print(output_seq2vec.sum())

    # batch_size x 2400
    output_seq2vec2 = uniskip(input)

    # # batch_size x seq_len x 2400
    # output_seq2seq = uniskip(input)