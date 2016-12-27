local npy4th = require 'npy4th'
local skipthoughts = require '../skipthoughts'

local cmd = torch.CmdLine()
cmd:option('-dirname', '/local/cadene/data/skip-thoughts/final', '')
local config = cmd:parse(arg)

if not paths.dirp('test/data') then
   -- it may take one hour...
   -- be sure you set the right path in skip-thoughts/skipthoughts.py
   os.execute('python test/createFeatures.py')
end

local tester = torch.Tester()
local mytest = torch.TestSuite()

-----------------------------------------------
-- OneWord, NormFalse

function mytest.UniSkipSeq2Vec_OneWord_NormFalse_EosFalse()
   local vocab = {'robots', 'are', 'cool'}
   local inputs = {
      torch.Tensor{1}
   }
   local skip = skipthoughts.createUniSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkipSeq2Vec_OneWord_NormFalse_EosFalse()
   local vocab = {'are', 'robots', 'cool'}
   local inputs = {
      torch.Tensor{2}
   }
   local skip = skipthoughts.createBiSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_OneWord_NormFalse_EosFalse()
   local vocab = {'robots'}
   local inputs = {
      torch.Tensor{1}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-----------------------------------------------
-- OneWord, NormTrue

function mytest.UniSkipSeq2Vec_OneWord_NormTrue_EosFalse()
   local vocab = {'robots'}
   local inputs = {
      torch.Tensor{1}
   }
   local skip = skipthoughts.createUniSkipSeq2Vec(vocab, config.dirname, true)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkipSeq2Vec_OneWord_NormTrue_EosFalse()
   local vocab = {'robots'}
   local inputs = {
      torch.Tensor{1}
   }
   local skip = skipthoughts.createBiSkipSeq2Vec(vocab, config.dirname, true)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_OneWord_NormTrue_EosFalse()
   local vocab = {'robots'}
   local inputs = {
      torch.Tensor{1}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname, true)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-----------------------------------------------
-- One Word, End of Sequence char

function mytest.UniSkipSeq2Vec_OneWord_NormFalse_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createUniSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkipSeq2Vec_OneWord_NormFalse_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createBiSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_OneWord_NormFalse_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-----------------------------------------------
-- One Word, End of Sequence char, NormTrue

function mytest.UniSkipSeq2Vec_OneWord_NormTrue_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createUniSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkipSeq2Vec_OneWord_NormTrue_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createBiSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_OneWord_NormTrue_EosTrue()
   local vocab = {'robots', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-----------------------------------------------
-- Multiple Words

function mytest.UniSkipSeq2Vec_NormFalse_EosFalse()
   local vocab = {'robots', 'are', 'cool'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3}
   }
   local skip = skipthoughts.createUniSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkipSeq2Vec_NormFalse_EosFalse()
   local vocab = {'robots', 'are', 'cool'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3}
   }
   local skip = skipthoughts.createBiSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_NormFalse_EosFalse()
   local vocab = {'robots', 'are', 'cool'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- -----------------------------------------------
-- -- Multiple words, Norm True|False, Eos True|False

function mytest.CombineSkipSeq2Vec_NormTrue_EosFalse()
   local vocab = {'robots', 'are', 'cool'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname, true)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normTrue_eosFalse.npy')
   print(torch.dist(features, outputs), features[1][1], outputs[1][1])
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_NormFalse_EosTrue()
   local vocab = {'robots', 'are', 'cool', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3},
      torch.Tensor{4}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosTrue.npy')
   print(torch.dist(features, outputs), features[1][1], outputs[1][1])
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.CombineSkipSeq2Vec_NormTrue_EosTrue()
   local vocab = {'robots', 'are', 'cool', '<eos>'}
   local inputs = {
      torch.Tensor{1},
      torch.Tensor{2},
      torch.Tensor{3},
      torch.Tensor{4}
   }
   local skip = skipthoughts.createCombineSkipSeq2Vec(vocab, config.dirname, true)
   local outputs = skip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normTrue_eosTrue.npy')
   print(torch.dist(features, outputs), features[1][1], outputs[1][1])
   tester:eq(features, outputs, 0.0001,
      "features and outputs should be approximately equal")
end

tester:add(mytest)

-- tester:add(mytest.UniSkipSeq2Vec_NormFalse_EosFalse,
--                  'UniSkipSeq2Vec_NormFalse_EosFalse')

-- tester:add(mytest.UniSkipSeq2Vec_OneWord_NormTrue_EosFalse,
--                  'UniSkipSeq2Vec_OneWord_NormTrue_EosFalse')
-- tester:add(mytest.UniSkipSeq2Vec_OneWord_NormFalse_EosTrue,
--                  'UniSkipSeq2Vec_OneWord_NormFalse_EosTrue')
-- tester:add(mytest.UniSkipSeq2Vec_OneWord_NormTrue_EosTrue,
--                  'UniSkipSeq2Vec_OneWord_NormTrue_EosTrue')

-- tester:add(mytest.BiSkipSeq2Vec_OneWord_NormFalse_EosFalse,
--                  'BiSkipSeq2Vec_OneWord_NormFalse_EosFalse')
-- tester:add(mytest.BiSkipSeq2Vec_NormFalse_EosFalse,
--                  'BiSkipSeq2Vec_NormFalse_EosFalse')

-- tester:add(mytest.BiSkipSeq2Vec_OneWord_NormFalse_EosTrue,
--                  'BiSkipSeq2Vec_OneWord_NormFalse_EosTrue')

-- tester:add(mytest.CombineSkipSeq2Vec_NormFalse_EosFalse,
--                  'CombineSkipSeq2Vec_NormFalse_EosFalse')

-- tester:add(mytest.CombineSkipSeq2Vec_NormFalse_EosFalse,
--                  'CombineSkipSeq2Vec_NormFalse_EosFalse')
-- tester:add(mytest.CombineSkipSeq2Vec_NormTrue_EosFalse,
--                  'CombineSkipSeq2Vec_NormTrue_EosFalse')
-- tester:add(mytest.CombineSkipSeq2Vec_NormTrue_EosTrue,
--                  'CombineSkipSeq2Vec_NormTrue_EosTrue')

tester:run()
