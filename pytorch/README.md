# Skip-Thoughts.torch for Pytorcb

*Skip-Thoughts.torch* is a lightweight porting of [skip-thought pretrained models from Theano](https://github.com/ryankiros/skip-thoughts) to Pytorch.

## Installation

1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)

### Install from pip

3. `pip install skipthoughts`

### Install from repo

3. `git clone https://github.com/Cadene/skip-thoughts.torch.git`
4. `cd skip-thoughts.torch/pytorch`
5. `python setup.py install`


### Available pretrained models

#### UniSkip

It uses the `nn.GRU` layer from torch with the cudnn backend. It is the fastest implementation, but the dropout is sampled after each time-step in the cudnn implementation... (equals bad regularization)

#### DropUniSkip

It uses the `nn.GRUCell` layer from torch with the cudnn backend. It is slightly slower than UniSkip, however the dropout is sampled once for all time-steps in a sequence (good regularization).

#### BayesianUniSkip

It uses a custom GRU layer with a torch backend. It is at least two times slower than UniSkip, however the dropout is sampled once for all time-steps for each Linear (best regularization).

#### BiSkip

Equivalent to UniSkip, but with a bi-sequential GRU.

### Quick example

```python
import torch
from torch.autograd import Variable
import sys
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip

dir_st = 'data/skip-thoughts'
vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
uniskip = UniSkip(dir_st, vocab)

input = Variable(torch.LongTensor([
    [1,2,3,4,0], # robots are very cool 0
    [6,2,3,4,5]  # bidibu are very cool <eos>
])) # <eos> token is optional
print(input.size()) # batch_size x seq_len

output_seq2vec = uniskip(input, lengths=[4,5])
print(output_seq2vec.size()) # batch_size x 2400

output_seq2seq = uniskip(input)
print(output_seq2seq.size()) # batch_size x seq_len x 2400
```