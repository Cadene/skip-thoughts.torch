require 'nn'
local skipthoughts = require 'skipthoughts'

local cmd = torch.CmdLine()
cmd:option('-dirname', '/local/cadene/data/skip-thoughts', '')
local config = cmd:parse(arg)

local dir_final = paths.concat(config.dirname, 'final')

local vocab = {'skipthoughts','vectors','hello','robot','cool','are'}

-- Inputs and outputs to be of shape seqlen x batchsize x featsize
-- That is a batch of two sentences must be as follow:
inputs = {
   torch.Tensor{0,1},
   torch.Tensor{4,2},
   torch.Tensor{6,6},
   torch.Tensor{5,5}
}
-- First sentence: {ZeroPadding, robot, are, cool}
-- Second sentence: {skipthoughts, vectors, are, cool}

--------------------------------------------
-- Skip Thoughts seq2seq models 
--------------------------------------------

-- Those models encode a sequence of words
-- to a sequence of features of the same size.

local uni_skip = skipthoughts.createUniSkip(vocab, dir_final)
local outputs = uni_skip:forward(inputs)
print(#outputs)     -- 4 (4 words)
print(#outputs[1])  -- 2, 2400 (2 sentences, 2400 features)

local bi_skip = skipthoughts.createBiSkip(vocab, dir_final)
local outputs = bi_skip:forward(inputs)
print(#outputs)     -- 4
print(#outputs[1])  -- 2, 2400

local cb_skip = skipthoughts.createCombineSkip(vocab, dir_final)
local outputs = cb_skip:forward(inputs)
print(#outputs)        -- 2 (2 models)
print(#outputs[1])     -- 4 (4 words)
print(#outputs[1][1])  -- 2, 2400

--------------------------------------------
-- Skip Thoughts seq2vec models 
--------------------------------------------

-- If you want to encode a sequence of words
-- to one features vector, you have to select
-- the last corresponding features of the sequence.
-- You can do it as follow:

local uni_skip_s2v = skipthoughts.createUniSkipSeq2Vec(vocab, dir_final)
local outputs = uni_skip_s2v:forward(inputs)
print(#outputs) -- 2, 2400

local bi_skip_s2v = skipthoughts.createBiSkipSeq2Vec(vocab, dir_final)
local outputs = bi_skip_s2v:forward(inputs)
print(#outputs) -- 2, 2400

local cb_skip_s2v = skipthoughts.createCombineSkipSeq2Vec(vocab, dir_final)
local outputs = cb_skip_s2v:forward(inputs)
print(#outputs) -- 2, 4800

