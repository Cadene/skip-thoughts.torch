# Skip-Thoughts.torch

*Skip-Thoughts.torch* is a binding from Theano to Torch7 allowing to use the skip-thought models.

The uni-skip model is made of a hashmap which map a word (from a dictionnary of 900,000 words) to its corresponding vector (620 dimensions) just as word2vec and a GRU which takes the latter and process the final skip-thought vector (2400 dimensions).

The bi-skip model is made of a different hashmap and two GRUs. The first one is a forward GRU which takes a vector (620 dimensions) and output a vector (1200 dimensions). The second one is a backward GRU which takes a vector (620 dimensions) and output a vector (1200 dimensions). The final skip-thought vector is the result of the concatenation of the two vectors.

## How to use in torch7 ?

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

The initial vocabulary is made of 900,000 words. You will certainly want to create a LookupTable which will map the smaller vocabulary of your dataset to their corresponding vectors.
```lua
TODO
```

## How to reproduce the binding ?

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
