# Skip-Thoughts.torch

*Skip-Thoughts.torch* is a lightweight porting of [skip-thought pretrained models from Theano](https://github.com/ryankiros/skip-thoughts) to Torch7 using the [rnn](https://github.com/Element-Research/rnn) library of Element-Research and [npy4th](https://github.com/htwaijry/npy4th).

## Using the pretrained models in Torch7

### Requirements

```
$ luarocks install tds  # for the hashmap
$ luarocks install rnn  # for the rnn utils
$ luarocks install --server=http://luarocks.org/dev skipthoughts
```

### Quick example

The skip-thoughts package enables you to download the pretrained torch7 hashmaps and GRUs, and also to cleanly set the pretrained skip-thoughts models. In fact, the initial vocabulary is made of 930,913 words (including the vocabulary of *word2vec*). That is why, it is preferable to create a `nn.LookupTableMaskZero` in order to map your smaller vocabulary to their corresponding vectors in an efficient and "fine-tunable" way. See an example bellow:

```lua
st = require 'skipthoughts'
vocab = {'skipthoughts', 'are', 'cool'}
inputs = torch.Tensor{{1,2,3}} -- batch x seq
-- Download and load pretrained models on the fly
uni_skip = st.createUniSkip(vocab, 'data')
print(uni_skip:forward(inputs):size()) -- batch x 2400
```

For further examples please refer to [torch/example.lua](https://github.com/Cadene/skip-thoughts.torch/blob/master/torch/example.lua) or [torch/test.lua](https://github.com/Cadene/skip-thoughts.torch/blob/master/torch/test.lua)


## Implementation details

### General information

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

### GRUST

We provide a new GRU called [GRUST](https://github.com/Cadene/skip-thoughts.torch/blob/master/torch/GRUST.lua) for "Gated Recurrent Unit for Skip-Toughts".
In fact, the authors of skip-thoughts models did not use the same implementation as in the `rnn` library.

The implementation of GRUST corresponds to the following algorithm:
```
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + r[t] .* W[hr->c](s[t−1]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
```
(with `.*` the element wise product)
Note: It is also the same implementation of GRU from pytorch.

Whereas, the implementation of GRU from the [rnn](https://github.com/Element-Research/rnn#rnn.GRU) package corresponds to the following algorithm:
```
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1] .* r[t]) + b[1->h])  (3)
```
(with `.*` the element wise product)

### MaskZeroCopy

We provide a new layer for the bi-skip and (thus) combine-skip models. In fact, the backward GRU may recieve inputs with right zero padding instead of the usual left zero padding. Thus, the MaskZeroCopy layer will copy the last outputs of the backward GRU when it sees vectors of zero, instead of replacing the actual content by zero (usual behaviour of MaskZero).

```
-- The input is "hello world" but our full model takes batchs of size 3 only
-- thus we need to add a 0 on the left (left zero padding).
-- The ouput of the GRU forward must be the result of precessing hello and then word (= features(hello,word)).
-- Wheras the output of the GRU backward must be the result of processing word, then hello (= features(word,hello)).
input = {0, hello, world}
reverse_input = {world, hello, 0}

-- GRU forward in bi-skip model

-- without MaskZero the final output will be features(0,hello,world)
GRU_fw:forward(input) = {features(0), features(0,hello), features(0,hello,world)} 

-- with MaskZero the final output will be features(hello,world)
GRU_fw:forward(input) = {0, features(hello), features(hello,world)} 

-- GRU backward in bi-skip model

-- without MaskZero the final output will be features(world,hello,0)
GRU_bw:forward(reverse_input) = {features(world), features(world,hello), features(world,hello,0)}

-- with MaskZero the final output will be 0
GRU_bw:forward(reverse_input) = {features(world), features(world,hello), 0}

-- with MaskZeroCopy the final output will be features(word,hello)
GRU_bw:forward(reverse_input) = {features(world), features(word,hello), features(word,hello)}
```


## (Optional) Recreating torch7 files

### Requirements

Lua/Torch7 and Python2.

```
$ luarocks install tds
$ luarocks install rnn
$ luarocks install npy4th
$ pip install numpy
$ pip install theano
$ git clone https://github.com/Cadene/skip-thoughts.torch.git
$ cd skip-thoughts.torch
$ git submodule update --init --recursive # download my fork in theano/skip-thoughts
```

### Hashmaps

Create `uni_hashmap.t7` and `bi_hashmap.t7` (both of type [tds.Hash](https://github.com/torch/tds#d--tdshashtbl)) in `data/final`:
```
$ th torch/create_hashmaps.lua -dirname data
```

### GRUs

Create `uni_gru.t7`, `bi_gru_fwd.t7` and `bi_gru_bwd.t7` (every three of type [GRUST](https://github.com/Cadene/skip-thoughts.torch/blob/master/torch/GRUST.lua)) in `data/final`:
```
$ th torch/create_grus.lua -dirname data
```

### Test

```
$ th torch/test.lua -dirname data
```

## Acknowledgment

Beside the whole deep learning community, we would like to specifically thanks:
- the authors of the original [paper](https://arxiv.org/abs/1506.06726) and [implementation](https://github.com/ryankiros/skip-thoughts),
- the authors of DPPnet who first propose a [porting](https://github.com/HyeonwooNoh/DPPnet),
- the authors of Multi Modal Residual Learning who also propose a [porting](https://github.com/jnhwkim/nips-mrn-vqa).
