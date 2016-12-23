require 'nn'
require 'rnn'
require 'tds'
-- local argcheck = require 'argcheck'

local skipthoughts = {}

skipthoughts.__download = function(dirname)
   os.execute('mkdir -p '..dirname)
   os.execute('wget http://uni_gru.t7 -P '..dirname)
   os.execute('wget http://uni_hashmap.t7 -P '..dirname)
   os.execute('wget http://bi_gru_fwd.t7 -P '..dirname)
   os.execute('wget http://bi_gru_bwd.t7 -P '..dirname)
   os.execute('wget http://bi_hashmap.t7 -P '..dirname)
end

skipthoughts.loadHashmap = function(dirname, mode)
   local mode = mode or 'uni'
   if not paths.dirp(dirname) then
      skipthoughts.__download(dirname)
   end
   return torch.load(paths.concat(dirname, mode..'_hashmap.t7'))
end

skipthoughts.createLookupTable = function(vocab, dirname, mode)
   local hashmap = skipthoughts.loadHashmap(dirname, mode)
   local lookup = nn.LookupTableMaskZero(#vocab, 620)
   for i=1, #vocab do
      if hashmap[vocab[i]] then
         print(hashmap[vocab[i]]:size())
	 print(hashmap[vocab[i]][1])
         print(lookup.weight[i+1][1])
	 lookup.weight[i+1]:copy(hashmap[vocab[i]]) -- i+1 because 1 is the 0 vector
         print(lookup.weight[i+1][1])
      else
         print('Warning '..vocab[i]..' not present in hashamp')
      end
   end
   return lookup
end

--------------------------------------------
-- Skip Thoughts seq2seq models 

skipthoughts.createUniSkip = function(vocab, dirname)
   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'uni')
   local gru = torch.load(paths.concat(dirname, 'uni_gru.t7')):trimZero(1)
   local skip = nn.Sequential()
      :add(lookup)
      --:add(nn.Normalize(2))
      :add(gru)
   return nn.Sequencer(skip)
end

skipthoughts.createBiSkip = function(vocab, dirname)
   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'bi')
   local gru_fwd = torch.load(paths.concat(dirname, 'bi_gru_fwd.t7'))
   local gru_bwd = torch.load(paths.concat(dirname, 'bi_gru_bwd.t7'))
   local skip = nn.Sequential()
      :add(nn.Sequencer(lookup))
      :add(nn.BiSequencer(gru_fwd, gru_bwd))
   return skip
end

skipthoughts.createCombineSkip = function(vocab, dirname)
   local uni_skip = skipthoughts.createUniSkip(vocab, dirname)
   local bi_skip = skipthoughts.createBiSkip(vocab, dirname)
   local skip = nn.ConcatTable()
      :add(uni_skip)
      :add(bi_skip)
   return skip
end

--------------------------------------------
-- Skip Thoughts seq2vec models 

skipthoughts.createUniSkipSeq2Vec = function(vocab, dirname)
   local skip = skipthoughts.createUniSkip(vocab, dirname)
   local skip_s2v = nn.Sequential()
      :add(skip)
      :add(nn.SelectTable(-1))
   return skip_s2v
end

skipthoughts.createBiSkipSeq2Vec = function(vocab, dirname)
   local skip = skipthoughts.createBiSkip(vocab, dirname)
   local skip_s2v = nn.Sequential()
      :add(skip)
      :add(nn.SelectTable(-1))
   return skip_s2v
end

skipthoughts.createCombineSkipSeq2Vec = function(vocab, dirname)
   local skip = skipthoughts.createCombineSkip(vocab, dirname)
   local uni_part = nn.Sequential()
   uni_part:add(nn.SelectTable(1)) -- takes uni-skip outputs
   uni_part:add(nn.SelectTable(-1)) -- last features in seq
   local bi_part = nn.Sequential()
   bi_part:add(nn.SelectTable(2)) -- takes bi-skip outputs
   bi_part:add(nn.SelectTable(-1))
   local parallel = nn.ConcatTable()
   parallel:add(uni_part)
   parallel:add(bi_part)
   local skip_s2v = nn.Sequential()
   skip_s2v:add(skip)
   skip_s2v:add(parallel)
   skip_s2v:add(nn.JoinTable(2))
   return skip_s2v
end

return skipthoughts
