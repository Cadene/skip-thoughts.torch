# Skip-Thoughts.torch

*Skip-Thoughts.torch* is a lightweight porting of skip-thought pretrained models from Theano to Torch7 using the beautiful [rnn](https://github.com/Element-Research/rnn) library of Element-Research and [npy4th](https://github.com/htwaijry/npy4th).

The **uni-skip model** is made of:
- a hashmap which, just as word2vec, map a word (from a dictionnary of 930,913 words) to its corresponding vector (620 dimensions),
- a GRU which takes as input the latter vector and process the final skip-thought vector (2400 dimensions).

The **bi-skip model** is made of:
- a different hashmap (but same dictionnary),
- a first GRU (forward) which takes a vector (620 dimensions) and output a vector (1200 dimensions),
- a second GRU (backward) which takes the same vector (620 dimensions) and output a vector (1200 dimensions).
The final skip-thought vector is the result of the concatenation of the two vectors (2400 dimensions).

The **combine-skip model** outputs the concatenation of both models output vectors (4800 dimensions).

Finally, once those pretrained models are set to take as input a sequence of words (notably by using the [nn.Sequencer](https://github.com/Element-Research/rnn#sequencer) and [nn.BiSequencer](https://github.com/Element-Research/rnn#bisequencer)), they can be used to compute a sequence of features of the same size (**seq2seq**) or a features vector (**seq2vec**).

## How to use the pretrained models ?

Install the requirements:
```
$ luarocks install tds  # for the hashmap
$ luarocks install rnn  # for the rnn utils
$ git clone https://github.com/Cadene/grust.torch
$ cd grust.torch
$ luarocks make rocks/grust-scm-1.rockspec
```

/!\ We need [GRUST](https://github.com/Cadene/grust.torch) because the authors of skip-thoughts models did not used the same implementation (please refer to the README of grust.torch).

We provide [skipthoughts.lua](https://github.com/Cadene/skip-thoughts.torch/blob/master/skipthoughts.lua), a library to easly use the pretrained skip-thoughts models.
The latter enables you to download the pretrained torch7 hashmaps and GRUs compressed in a [zip file]() hosted on google drive, and also to cleanly set the pretrained skip-thoughts models. In fact, the initial vocabulary is made of 930,913 words (including the vocabulary of *word2vec*). That is why, it is preferable to create a `nn.LookupTableMaskZero` in order to map your smaller vocabulary to their corresponding vectors in an efficient and "fine-tunable" way. See an example bellow:

```lua
st = require 'skipthoughts'
vocab = {'skipthoughts', 'are', 'cool'}
inputs = torch.Tensor{{1,2,3}} -- batch x seq
dirname = 'data'
-- Download and load pretrained models on the fly
uni_skip = st.createUniSkip(vocab, 'dirname')
print(uni_skip:forward(inputs):size()) -- batch x 2400
```

For further examples please refer to [test/test.lua](https://github.com/Cadene/skip-thoughts.torch/blob/master/test/test.lua).

## How to recreate the torch7 files ?

Install the requirements:
```
$ luarocks install tds
$ luarocks install rnn
$ luarocks install npy4th
$ pip install numpy
$ pip install theano
```

Create `uni_hashmap.t7` and `bi_hashmap.t7` (both of type [tds.Hash](https://github.com/torch/tds#d--tdshashtbl)) in `data/final`:
```
$ th create_hashmaps.lua -dirname data
```

Create `uni_gru.t7`, `bi_gru_fwd.t7` and `bi_gru_bwd.t7` (every three of type [GRUST](https://github.com/Cadene/grust.torch)) in `data/final`:
```
$ th create_grus.lua -dirname data
```

## Acknowledgment

Beside the wall deep learning community, we would like to specifically thanks:
- the authors of the original [paper](https://arxiv.org/abs/1506.06726) and [implementation](https://github.com/ryankiros/skip-thoughts),
- the authors of DPPnet who first propose a [porting](https://github.com/HyeonwooNoh/DPPnet),
- the authors of Multi Modal Residual Learning who also propose a [porting](https://github.com/jnhwkim/nips-mrn-vqa).
