local npy4th = require 'npy4th'
local skipthoughts = require '../skipthoughts'

local cmd = torch.CmdLine()
-- cmd:option('-dirname', '/local/cadene/data/skip-thoughts/final', '')
cmd:option('-dirname', 'data', '')
local config = cmd:parse(arg)

if not paths.dirp('test/data') then
   -- it may take one hour...
   -- be sure you set the right path in skip-thoughts/skipthoughts.py
   os.execute('python theano/dump_features.py')
end

local tester = torch.Tester()
local mytest = torch.TestSuite()

local vocab = {'robots', 'are', 'very', 'cool', '<eos>'}

local dropout = 0.25

local uniskip = skipthoughts.createUniSkip(vocab, config.dirname, dropout)
uniskip:evaluate()

local biskip = skipthoughts.createBiSkip(vocab, config.dirname, dropout)
biskip:evaluate()

-----------------------------------------------
-- OneWord, NormFalse

function mytest.UniSkip_OneWord_NormFalse_EosFalse()
   local inputs = torch.Tensor{{1}}
   local outputs = uniskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_OneWord_NormFalse_EosFalse()
   local inputs = torch.Tensor{{1}}
   local outputs = biskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- function mytest.CombineSkip_OneWord_NormFalse_EosFalse()
--    local inputs = torch.Tensor{{1}}
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-----------------------------------------------
-- OneWord, NormTrue

function mytest.UniSkip_OneWord_NormTrue_EosFalse()
   local inputs = torch.Tensor{{1}}
   local outputs = nn.Normalize(2):forward(uniskip:forward(inputs)):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_OneWord_NormTrue_EosFalse()
   local inputs = torch.Tensor{{1}}
   local outputs = nn.Normalize(2):forward(biskip:forward(inputs)):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- function mytest.CombineSkip_OneWord_NormTrue_EosFalse()
--    local vocab = {'robots'}
--    local inputs = torch.Tensor{{1}}
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname, true)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosFalse.npy')
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-----------------------------------------------
-- OneWord, NormTrue, ZeroPadding

function mytest.UniSkip_OneWord_NormFalse_EosFalse_ZeroPadding()
   local inputs = torch.Tensor{{0,0,1}}
   local outputs = uniskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_OneWord_NormFalse_EosFalse_ZeroPadding()
   local inputs = torch.Tensor{{0,0,1}}
   local outputs = biskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-----------------------------------------------
-- One Word, End of Sequence char

function mytest.UniSkip_OneWord_NormFalse_EosTrue()
   local inputs = torch.Tensor{{0,1,5}}
   local outputs = uniskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- function mytest.BiSkip_OneWord_NormFalse_EosTrue()
--    local vocab = {'robots', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2}
--    }
--    local skip = skipthoughts.createBiSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
--    tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-- function mytest.CombineSkip_OneWord_NormFalse_EosTrue()
--    local vocab = {'robots', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normFalse_eosTrue.npy')
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-----------------------------------------------
-- One Word, End of Sequence char, NormTrue

function mytest.UniSkip_OneWord_NormTrue_EosTrue()
   local inputs = torch.Tensor{{0,1,5}}
   local outputs = nn.Normalize(2):forward(uniskip:forward(inputs)):float()
   local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- function mytest.BiSkip_OneWord_NormTrue_EosTrue()
--    local vocab = {'robots', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2}
--    }
--    local skip = skipthoughts.createBiSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
--    tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-- function mytest.CombineSkip_OneWord_NormTrue_EosTrue()
--    local vocab = {'robots', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_oneWord_normTrue_eosTrue.npy')
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-----------------------------------------------
-- Multiple Words

function mytest.UniSkip_NormFalse_EosFalse()
   local inputs = torch.Tensor{{0,1,2,4}}
   local outputs = uniskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 1, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_NormFalse_EosFalse_forward()
   local inputs = torch.Tensor{{0,1,2,4}}
   local outputs = biskip:forward(inputs):float():narrow(2, 1, 1200)
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 1200), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_NormFalse_EosFalse_backward()
   --local inputs = torch.Tensor{{0,1,2,4}}
   local inputs = torch.Tensor{{1,2,4}}
   local outputs = biskip:forward(inputs):float():narrow(2, 1201, 1200)
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 3601, 1200), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

function mytest.BiSkip_NormFalse_EosFalse()
   local inputs = torch.Tensor{{0,1,2,4}}
   local outputs = biskip:forward(inputs):float()
   local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
   tester:eq(features:narrow(2, 2401, 2400), outputs, 0.0001,
      "features and outputs should be approximately equal")
end

-- function mytest.CombineSkip_NormFalse_EosFalse()
--    local vocab = {'robots', 'are', 'cool'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2},
--       torch.Tensor{3}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_normFalse_eosFalse.npy')
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-- -----------------------------------------------
-- -- Multiple words, Norm True|False, Eos True|False

-- function mytest.CombineSkip_NormTrue_EosFalse()
--    local vocab = {'robots', 'are', 'cool'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2},
--       torch.Tensor{3}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname, true)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_normTrue_eosFalse.npy')
--    print(torch.dist(features, outputs), features[1][1], outputs[1][1])
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-- function mytest.CombineSkip_NormFalse_EosTrue()
--    local vocab = {'robots', 'are', 'cool', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2},
--       torch.Tensor{3},
--       torch.Tensor{4}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_normFalse_eosTrue.npy')
--    print(torch.dist(features, outputs), features[1][1], outputs[1][1])
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

-- function mytest.CombineSkip_NormTrue_EosTrue()
--    local vocab = {'robots', 'are', 'cool', '<eos>'}
--    local inputs = {
--       torch.Tensor{1},
--       torch.Tensor{2},
--       torch.Tensor{3},
--       torch.Tensor{4}
--    }
--    local skip = skipthoughts.createCombineSkip(vocab, config.dirname, true)
--    local outputs = skip:forward(inputs):float()
--    local features = npy4th.loadnpy('test/data/features_normTrue_eosTrue.npy')
--    print(torch.dist(features, outputs), features[1][1], outputs[1][1])
--    tester:eq(features, outputs, 0.0001,
--       "features and outputs should be approximately equal")
-- end

--tester:add(mytest)

if uniskip then
   tester:add(mytest.UniSkip_NormFalse_EosFalse,
                    'UniSkip_NormFalse_EosFalse')
   tester:add(mytest.UniSkip_OneWord_NormFalse_EosFalse_ZeroPadding,
                     'UniSkip_OneWord_NormFalse_EosFalse_ZeroPadding')
   tester:add(mytest.UniSkip_OneWord_NormFalse_EosFalse,
                     'UniSkip_OneWord_NormFalse_EosFalse')
   tester:add(mytest.UniSkip_OneWord_NormTrue_EosFalse,
                    'UniSkip_OneWord_NormTrue_EosFalse')
   tester:add(mytest.UniSkip_OneWord_NormFalse_EosTrue,
                    'UniSkip_OneWord_NormFalse_EosTrue')
   tester:add(mytest.UniSkip_OneWord_NormTrue_EosTrue,
                    'UniSkip_OneWord_NormTrue_EosTrue')
end

if biskip then
   tester:add(mytest.BiSkip_OneWord_NormFalse_EosFalse,
                    'BiSkip_OneWord_NormFalse_EosFalse')
   tester:add(mytest.BiSkip_OneWord_NormFalse_EosFalse_ZeroPadding,
                    'BiSkip_OneWord_NormFalse_EosFalse_ZeroPadding')
   tester:add(mytest.BiSkip_NormFalse_EosFalse_forward,
                    'BiSkip_NormFalse_EosFalse_forward')
   tester:add(mytest.BiSkip_NormFalse_EosFalse_backward,
                    'BiSkip_NormFalse_EosFalse_backward')
   tester:add(mytest.BiSkip_NormFalse_EosFalse,
                    'BiSkip_NormFalse_EosFalse')
end

-- tester:add(mytest.BiSkip_OneWord_NormFalse_EosTrue,
--                  'BiSkip_OneWord_NormFalse_EosTrue')

-- tester:add(mytest.CombineSkip_NormFalse_EosFalse,
--                  'CombineSkip_NormFalse_EosFalse')

-- tester:add(mytest.CombineSkip_NormFalse_EosFalse,
--                  'CombineSkip_NormFalse_EosFalse')
-- tester:add(mytest.CombineSkip_NormTrue_EosFalse,
--                  'CombineSkip_NormTrue_EosFalse')
-- tester:add(mytest.CombineSkip_NormTrue_EosTrue,
--                  'CombineSkip_NormTrue_EosTrue')

tester:run()
