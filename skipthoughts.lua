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
         lookup.weight[i]:copy(hashmap[vocab[i]])
      else
         print('Warning '..vocab[i]..' not present in hashamp')
      end
   end
   return lookup
end

skipthoughts.createUniSkip = function(vocab, dirname)
   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'uni')
   local gru = torch.load(paths.concat(dirname, 'uni_gru.t7'))
   local skip = nn.Sequential()
      :add(lookup)
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

return skipthoughts