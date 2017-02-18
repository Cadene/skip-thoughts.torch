package = "skipthoughts"
version = "scm-1"

source = {
   url = "git://github.com/Cadene/skip-thoughts.torch",
   tag = "master"
}

description = {
   summary = "Skip-thoughts models for Torch7",
   detailed = [[
Porting of pretrained skip-thoughts models from Theano to Torch7.
   ]],
   homepage = "https://github.com/Cadene/skip-thoughts.torch",
   license = "MIT License"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "tds >= 1.0",
   "rnn >= 1.0",
}

build = {
   type = "builtin",
   modules = {
      ["skipthoughts.init"] = "torch/init.lua",
      ["skipthoughts.MaskZeroCopy"] = "torch/MaskZeroCopy.lua",
      ["skipthoughts.GRUST"] = "torch/GRUST.lua",
      ["skipthoughts.skipthoughts"] = "torch/skipthoughts.lua",
   }
}