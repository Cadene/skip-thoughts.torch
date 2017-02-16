local npy4th = require 'npy4th'
local tds = require 'tds'

function download(dirraw)
   os.execute('mkdir -p ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt -P ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/utable.npy -P ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/btable.npy -P ' .. dirraw)
end

function load_dico(path)
   local dico = tds.Hash()
   local count = 1
   if not paths.filep(path) then
      return false
   end
   local f = io.open(path, 'rb')
   for line in f:lines() do
      dico[line] = count
      count = count + 1
   end
   return dico
end

function create_hashmap(dico, tensor)
   local hashmap = tds.Hash()
   for word, id in pairs(dico) do
      hashmap[word] = tensor[id]
   end
   return hashmap
end

---------------------------------------------
-- Arguments
---------------------------------------------

local cmd = torch.CmdLine()
-- cmd:option('-dirname', '/local/cadene/data/skip-thoughts', '')
cmd:option('-dirname', 'data', '')
local config = cmd:parse(arg)

---------------------------------------------
-- Path to files
---------------------------------------------

dir_raw = paths.concat(config.dirname, 'raw')
path_dico = paths.concat(dir_raw, 'dictionary.txt')
-- path_utable = paths.concat(dir_raw, 'utable.npy')
-- path_utable = paths.concat(dir_raw, 'btable.npy')

dir_interim = paths.concat(config.dirname, 'interim')
path_utable_npy = paths.concat(dir_interim, 'utable.npy')
path_btable_npy = paths.concat(dir_interim, 'btable.npy')

dir_processed = paths.concat(config.dirname, 'processed')
path_dico_t7 = paths.concat(dir_processed, 'dictionary.t7')
path_utable_t7 = paths.concat(dir_processed, 'utable.t7')
path_btable_t7 = paths.concat(dir_processed, 'btable.t7')

dir_final = paths.concat(config.dirname, 'final')
path_uhashmap_t7 = paths.concat(dir_final, 'uni_hashmap.t7')
path_bhashmap_t7 = paths.concat(dir_final, 'bi_hashmap.t7')

---------------------------------------------
-- Download and format npy
---------------------------------------------

if not paths.dirp(dir_raw) then
   download(dir_raw)
   os.execute('python theano/format_hashmaps.py --dirname '..config.dirname)
end

---------------------------------------------
-- Load to torch7
---------------------------------------------

local dico = load_dico(path_dico)
local utable = npy4th.loadnpy(path_utable_npy)
local btable = npy4th.loadnpy(path_btable_npy)

---------------------------------------------
-- Create hashmaps
---------------------------------------------

local uhashmap = create_hashmap(dico, utable)
local bhashmap = create_hashmap(dico, btable)

os.execute('mkdir -p '..dir_final)
torch.save(path_uhashmap_t7, uhashmap)
torch.save(path_bhashmap_t7, bhashmap)






