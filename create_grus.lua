require 'nn'
require 'rnn'
local npy4th = require 'npy4th'

function download(dirraw)
   os.execute('git clone https://github.com/ryankiros/skip-thoughts.git')
   os.execute('mkdir -p ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz -P ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl -P ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz -P ' .. dirraw)
   os.execute('wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl -P ' .. dirraw)
end

function create_gru(inputSize, outputSize, params)
   local gru = nn.GRU(inputSize, outputSize)
   local gru_params = gru:parameters()
   gru_params[1]:copy(params.W:t())
   gru_params[2]:copy(params.b)
   gru_params[3]:copy(params.U:t())
   gru_params[4]:copy(params.Wx:t())
   gru_params[5]:copy(params.bx)
   gru_params[6]:copy(params.Ux:t())
   return gru
end

---------------------------------------------
-- Arguments and paths
---------------------------------------------

local cmd = torch.CmdLine()
cmd:option('-dirname', '/local/cadene/data/skip-thoughts', '')
local config = cmd:parse(arg)

local dir_raw = paths.concat(config.dirname, 'raw')
local dir_interim = paths.concat(config.dirname, 'interim')

local dir_final = paths.concat(config.dirname, 'final')
local path_uni_gru = paths.concat(dir_final, 'uni_gru.t7')
local path_bi_gru_fwd = paths.concat(dir_final, 'bi_gru_fwd.t7')
local path_bi_gru_bwd = paths.concat(dir_final, 'bi_gru_bwd.t7')
os.execute('mkdir -p '..dir_final)

---------------------------------------------
-- Download and format parameters with theano
---------------------------------------------

if not paths.dirp(dir_raw) or
   not paths.filep(paths.concat(dir_raw, 'bi_skip.npz.pkl')) then
   download(dir_raw)
end
if not paths.dirp(dir_interim) then
   os.execute('python format_params.py --dirname '..config.dirname)
end

---------------------------------------------
-- Create uni-GRU
---------------------------------------------

print('Load uni-GRU params')
local uparams = {}
uparams.U  = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_U.npy'))
uparams.Ux = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_Ux.npy'))
uparams.W  = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_W.npy'))
uparams.b  = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_b.npy'))
uparams.Wx = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_Wx.npy'))
uparams.bx = npy4th.loadnpy(paths.concat(dir_interim, 'uparams_encoder_bx.npy'))

local inputSize = 620
local outputSize = 2400
local uni_gru = create_gru(inputSize, outputSize, uparams)
torch.save(path_uni_gru, uni_gru)

---------------------------------------------
-- Create bi-GRU
---------------------------------------------

print('Load bi-GRU params')
local bparams = {}
bparams.U  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_U.npy'))
bparams.Ux = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_Ux.npy'))
bparams.W  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_W.npy'))
bparams.b  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_b.npy'))
bparams.Wx = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_Wx.npy'))
bparams.bx = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_bx.npy'))

local inputSize = 620
local outputSize = 1200
local bi_gru_fwd = create_gru(inputSize, outputSize, bparams)
torch.save(path_bi_gru_fwd, bi_gru_fwd)

print('Load bi-GRU reverse params')
local bparams_r = {}
bparams_r.U  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_U.npy'))
bparams_r.Ux = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_Ux.npy'))
bparams_r.W  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_W.npy'))
bparams_r.b  = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_b.npy'))
bparams_r.Wx = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_Wx.npy'))
bparams_r.bx = npy4th.loadnpy(paths.concat(dir_interim, 'bparams_encoder_r_bx.npy'))

local inputSize = 620
local outputSize = 1200
local bi_gru_bwd = create_gru(inputSize, outputSize, bparams_r)
torch.save(path_bi_gru_bwd, bi_gru_bwd)


