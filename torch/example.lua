require 'nn'
local skipthoughts = require 'skipthoughts'

local cmd = torch.CmdLine()
-- cmd:option('-dirname', '/local/cadene/data/skip-thoughts/final', '')
cmd:option('-dirname', 'data', '')
local config = cmd:parse(arg)

local dirfinal = paths.concat(config.dirname, 'final')

local vocab = {'robots', 'are', 'very', 'cool', '<eos>'}

-- Inputs and outputs to be of shape seqlen x batchsize x featsize
-- That is a batch of two sentences must be as follow:
local inputs = torch.Tensor{
   {1,3,4,0},
   --{0,1,3,4,5,0},
   --{1,2,3,4,5,0}
}
-- First sentence: {ZeroPadding, ZeroPadding, robots, are, cool}
-- Second sentence: {ZeroPadding, robots, are, cool, <eos>}
-- Third sentence: {robots, are, very, cool, <eos>}

--------------------------------------------
-- Skip Thoughts seq2vec models 
--------------------------------------------

-- If you want to encode a sequence of words
-- to one features vector, you have to select
-- the last corresponding features of the sequence.
-- You can do it as follow:
local dropout = 0.25
local uni_skip = skipthoughts.createUniSkip(vocab, dirfinal, dropout)
local outputs = uni_skip:forward(inputs)
print(#outputs) -- 3, 2400

O = outputs


