require 'nn'
require 'rnn'
require 'GRUST'
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
	      lookup.weight[i+1]:copy(hashmap[vocab[i]]) -- i+1 because 1 is the 0 vector
      else
         print('Warning '..vocab[i]..' not present in hashamp')
      end
   end
   return lookup
end

--------------------------------------------
-- Skip Thoughts seq2vec models 

skipthoughts.createUniSkip = function(vocab, dirname, dropout, norm)

   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'uni')
   local gru = torch.load(paths.concat(dirname, 'uni_gru.t7'))
   
   if dropout then
      local bgru = nn.GRUST(gru.inputSize, gru.outputSize, false, dropout, true)
      bgru:migrate(gru:parameters())
      gru = bgru
   end

   gru:trimZero(1)
   
   local seq_gru = nn.Sequencer(gru)
   --seq_gru = nn.TrimZero(seq_gru, 2)

   local skip = nn.Sequential()
   skip:add(lookup)
   -- skip:add(nn.SplitTable(2))
   skip:add(nn.Transpose({1,2}))
   skip:add(seq_gru)
   skip:add(nn.Transpose({2,1}))
   skip:add(nn.SplitTable(2))
   skip:add(nn.SelectTable(-1))
   if norm then
      skip:add(nn.Normalize(2))
   end

   return skip
end




skipthoughts.createBiSkip = function(vocab, dirname, norm)
   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'bi')
   local gru_fwd = torch.load(paths.concat(dirname, 'bi_gru_fwd.t7'))
   local gru_bwd = torch.load(paths.concat(dirname, 'bi_gru_bwd.t7'))
   gru_fwd:trimZero(1)
   gru_bwd:trimZero(1)

   local merge = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(2))
         :add(nn.SelectTable(1)))
      :add(nn.JoinTable(1,1))

   local skip = nn.Sequential()
   skip:add(nn.TrimZero(lookup,1))
   skip:add(nn.SplitTable(2))
   skip:add(nn.BiSequencer(gru_bwd, gru_fwd, merge))
   skip:add(nn.SelectTable(-1))
   if norm then
      skip:add(nn.Normalize(2))
   end

   return skip
end
















-- skipthoughts.createBiSkip = function(vocab, dirname)

--    local lookup = skipthoughts.createLookupTable(vocab, dirname, 'bi')
--    local gru_fwd = torch.load(paths.concat(dirname, 'bi_gru_fwd.t7'))
--    local gru_bwd = torch.load(paths.concat(dirname, 'bi_gru_bwd.t7'))
--    gru_fwd.batchfirst = true
--    gru_bwd.batchfirst = true

--    local merge = nn.Sequential()
--       :add(nn.ConcatTable()
--          :add(nn.SelectTable(2))
--          :add(nn.SelectTable(1)))
--       :add(nn.JoinTable(1,1))

--    local skip = nn.Sequential()
--       :add(lookup)
--       :add(nn.BiSequencer(gru_bwd, gru_fwd, merge))

--    return skip
-- end

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

skipthoughts.createUniSkipSeq2Vec = function(vocab, dirname, norm)
   local skip = skipthoughts.createUniSkip(vocab, dirname)
   --skip:add(nn.SplitTable(1))
   skip:add(nn.SelectTable(-1))
   if norm then
      skip:add(nn.Normalize(2))
   end
   return skip
end

skipthoughts.createBiSkipSeq2Vec = function(vocab, dirname, norm)
   local skip = skipthoughts.createBiSkip(vocab, dirname)
   --skip:add(nn.SplitTable(1))
   skip:add(nn.SelectTable(-1))
   if norm then
      skip:add(nn.Normalize(2))
   end
   return skip
end

skipthoughts.createCombineSkipSeq2Vec = function(vocab, dirname, norm)
   local skip = skipthoughts.createCombineSkip(vocab, dirname, norm)
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
   if norm then
      local parallel_norm = nn.ParallelTable()
      parallel_norm:add(nn.Normalize(2))
      parallel_norm:add(nn.Normalize(2))
      skip_s2v:add(parallel_norm)
   end
   skip_s2v:add(nn.JoinTable(2))
   return skip_s2v
end

skipthoughts.createUniSkip_mrn = function(vocab, dirname)
   local lookup = skipthoughts.createLookupTable(vocab, dirname, 'uni')
   local gru_mrn = torch.load(paths.concat(dirname, 'uni_gru_mrn.t7'))
   local gru = nn.GRU(gru_mrn.inputSize, gru_mrn.outputSize)
   local gru_mrn_params = gru_mrn:parameters()
   local gru_params = gru:parameters()
   gru_params[1]:copy(gru_mrn_params[1])
   gru_params[2]:copy(gru_mrn_params[2])
   gru_params[3]:copy(gru_mrn_params[3])
   gru_params[4]:copy(gru_mrn_params[4])
   gru_params[5]:copy(gru_mrn_params[5])
   gru_params[6]:copy(gru_mrn_params[6])
   local skip = nn.Sequential()
   skip:add(nn.Sequencer(lookup))
   skip:add(nn.Sequencer(gru))
   return skip
end

return skipthoughts
