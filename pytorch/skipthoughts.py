import os
import sys
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

from gru import BayesianGRU, GRU
from dropout import EmbeddingDropout, SequentialDropout

urls = {}
urls['dictionary'] = 'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt'
urls['utable']     = 'http://www.cs.toronto.edu/~rkiros/models/utable.npy'
urls['uni_skip']   = 'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz'
urls['btable']     = 'http://www.cs.toronto.edu/~rkiros/models/btable.npy'
urls['bi_skip']    = 'http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz'
    

class AbstractSkipThoughts(nn.Module):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractSkipThoughts, self).__init__()
        self.dir_st = dir_st
        self.vocab = vocab
        self.save = save
        self.dropout = dropout
        self.fixed_emb = fixed_emb
        # Module
        self.embedding = self._load_embedding()
        if fixed_emb:
            self.embedding.weight.requires_grad = False
        self.rnn = self._load_rnn()

    def _get_table_name(self):
        raise NotImplementedError

    def _get_skip_name(self):
        raise NotImplementedError

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
        table_name = self._get_table_name()
        path_params = os.path.join(self.dir_st, table_name+'.npy')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls[table_name], self.dir_st))
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
 
    def _load_rnn_params(self):
        skip_name = self._get_skip_name()
        path_params = os.path.join(self.dir_st, skip_name+'.npz')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls[skip_name], self.dir_st))
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
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1,
                                          embedding_dim=620,
                                          padding_idx=0, # -> first_dim = zeros
                                          sparse=False) 
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

    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i]-1].fill_(1)
        mask = Variable(mask)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, 2400)
        return x

    def _select_last_old(self, input, lengths):
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


###################################################################################
# UniSkip
###################################################################################

class AbstractUniSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _get_table_name(self):
        return 'utable'

    def _get_skip_name(self):
        return 'uni_skip'


class UniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(UniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)
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

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(DropUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)
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

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(BayesianUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

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
        max_length = max(lengths)
        x = self.embedding(input)
        x, hn = self.rnn(x, max_length=max_length) # seq2seq
        if lengths:
            x = self._select_last(x, lengths)
        return x


###################################################################################
# BiSkip
###################################################################################

class AbstractBiSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractBiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _get_table_name(self):
        return 'btable'

    def _get_skip_name(self):
        return 'bi_skip'


class BiSkip(AbstractBiSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(BiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)
        # Remove bias_ih_l0 (== zero all the time)
        # del self.gru._parameters['bias_hh_l0']
        # del self.gru._all_weights[0][3]

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620,
                          hidden_size=1200,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0']   = torch.zeros(3600) 
        s['bias_hh_l0']   = torch.zeros(3600) # must stay equal to 0
        s['weight_ih_l0'] = torch.zeros(3600, 620)
        s['weight_hh_l0'] = torch.zeros(3600, 1200)

        s['bias_ih_l0_reverse']   = torch.zeros(3600) 
        s['bias_hh_l0_reverse']   = torch.zeros(3600) # must stay equal to 0
        s['weight_ih_l0_reverse'] = torch.zeros(3600, 620)
        s['weight_hh_l0_reverse'] = torch.zeros(3600, 1200)
        
        s['weight_ih_l0'][:2400] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][2400:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:2400]   = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][2400:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:2400] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][2400:] = torch.from_numpy(p['encoder_Ux']).t()  

        s['weight_ih_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_W']).t()
        s['weight_ih_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Wx']).t()
        s['bias_ih_l0_reverse'][:2400]   = torch.from_numpy(p['encoder_r_b'])
        s['bias_ih_l0_reverse'][2400:]   = torch.from_numpy(p['encoder_r_bx'])
        s['weight_hh_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_U']).t()
        s['weight_hh_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Ux']).t() 
        return s

    def _argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        if lengths is None:
            lengths = self._process_lengths(input)
        sorted_lengths = sorted(lengths)
        sorted_lengths = sorted_lengths[::-1]
        idx = self._argsort(lengths)
        idx = idx[::-1] # decreasing order
        inverse_idx = self._argsort(idx)
        idx = Variable(torch.LongTensor(idx))
        inverse_idx = Variable(torch.LongTensor(inverse_idx))
        if input.data.is_cuda:
            idx = idx.cuda()
            inverse_idx = inverse_idx.cuda()
        x = torch.index_select(input, 0, idx)

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
        x, hn = self.rnn(x) # seq2seq
        hn = hn.transpose(0, 1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, 2 * hn.size(2))

        hn = torch.index_select(hn, 0, inverse_idx)
        return hn


if __name__ == '__main__':
    dir_st = '/home/cadene/data/skip-thoughts'
    vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
    #model = BayesianUniSkip(dir_st, vocab)
    model = BiSkip(dir_st, vocab)

    # batch_size x seq_len
    input = Variable(torch.LongTensor([
        [6,1,2,3,3,4,0],
        [6,1,2,3,3,4,5],
        [1,2,3,4,0,0,0]
    ]))
    print(input.size())
    model.eval()

    # batch_size x 2400
    output_seq2vec = model(input, lengths=[7,6,5])
    print(output_seq2vec)

    # batch_size x 2400
    output_seq2vec2 = model(input)

    # # batch_size x seq_len x 2400
    # output_seq2seq = model(input)