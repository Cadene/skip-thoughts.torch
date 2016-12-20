# Skip-Thoughts.torch

*Skip-Thoughts.torch* is a lightweight porting of skip-thought pretrained models from Theano to Torch7 using the beautiful *rnn* library of Element-Research and *npy4th*.

The uni-skip model is made of:
- a hashmap which, just as word2vec, map a word (from a dictionnary of 930,913 words) to its corresponding vector (620 dimensions),
- a GRU which takes as input the latter vector and process the final skip-thought vector (2400 dimensions).

The bi-skip model is made of:
- a different hashmap (but same dictionnary),
- a first GRU (forward) which takes a vector (620 dimensions) and output a vector (1200 dimensions),
- a second GRU (backward) which takes the same vector (620 dimensions) and output a vector (1200 dimensions).
The final skip-thought vector is the result of the concatenation of the two vectors (2400 dimensions).

## How to use the pretrained models ?

Install the requirements:
```
$ luarocks install tds  # for the hashmap
$ luarocks install rnn  # for the GRU
```

We provide several torch7 files to be able to easly load the skip-thought model of your choice:
```
$ wget TODO
$ wget TODO
```

The initial vocabulary is made of 930,913 words (including the vocabulary of *word2vec*). However, you would certainly want to create a LookupTable to map you smaller vocabulary to their corresponding vectors in an efficient and "fine-tunable" way.

```lua
st = require 'skipthoughts'
vocab = {'skipthoughts', 'are', 'cool'}
inputs = torch.Tensor{{1},{2},{3}}
uni_skip = st.createUniSkip(vocab)
bi_skip = st.createBiSkip(vocab)
cb_skip = st.createCombineSkip(vocab)
print(uni_skip:forward(inputs):size())
print(bi_skip:forward(inputs):size())
print(cb_skip:forward(inputs):size())
```

For further examples please refer to [example.lua](https://github.com/Cadene/skip-thoughts.torch/blob/master/example.lua).

## How to recreate the torch7 files ?

Install the requirements:
```
$ luarocks install tds
$ luarocks install rnn
$ luarocks install npy4th
$ pip install numpy
$ pip install theano
```

Create `uni_hashmap.t7` (of type tds.Hash) and `bi_hashmap.t7` (of type tds.Hash) in `data/final`:
```
$ th create_hashmaps.lua -dirname data
```

Create `uni_gru.t7` (of type nn.GRU) and `bi_gru.t7` (of type nn.BiSequencer) in `data/final`:
```
$ th create_grus.lua -dirname data
```

## Aknowledgment

Beside the wall deep learning community, we would like to specially thanks:
- [The authors of the original paper](https://arxiv.org/abs/1506.06726)
- [The authors of the original implementation](https://github.com/ryankiros/skip-thoughts)
- [The authors of DPPnet who first propose a porting](https://github.com/HyeonwooNoh/DPPnet)
- [The authors of Multi Modal Residual Learning who also propose a porting](https://github.com/jnhwkim/nips-mrn-vqa)
