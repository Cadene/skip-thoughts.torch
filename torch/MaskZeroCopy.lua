------------------------------------------------------------------------
--[[ MaskZeroCopy ]]--
-- Author: Remi Cadene

-- Decorator that zeroes the output rows of the encapsulated module
-- for commensurate input rows which are tensors of zeros
------------------------------------------------------------------------
local MaskZeroCopy, parent = torch.class("nn.MaskZeroCopy", "nn.Decorator")

function MaskZeroCopy:__init(module, nInputDim, silent, backward)
   parent.__init(self, module)
   assert(torch.isTypeOf(module, 'nn.Module'))
   if torch.isTypeOf(module, 'nn.AbstractRecurrent') and not silent then
      print("Warning : you are most likely using MaskZeroCopy the wrong way. "
      .."You should probably use AbstractRecurrent:maskZeroST() so that "
      .."it wraps the internal AbstractRecurrent.recurrentModule instead of "
      .."wrapping the AbstractRecurrent module itself.")
   end
   assert(torch.type(nInputDim) == 'number', 'Expecting nInputDim number at arg 1')
   self.nInputDim = nInputDim
   self._last_output = nil
   if self.backward then
      print("Warning : Backward is activated in MaskZeroCopy")
   else
      assert(false, 'not implemented')
   end
end

function MaskZeroCopy:recursiveGetFirst(input)
   if torch.type(input) == 'table' then
      return self:recursiveGetFirst(input[1])
   else
      assert(torch.isTensor(input))
      return input
   end
end

function MaskZeroCopy:recursiveMask(output, input, mask)
   local lastOutput = output:clone()

   --print(lastOutput:size())

   if torch.type(input) == 'table' then
      -- output = torch.type(output) == 'table' and output or {}
      -- for k,v in ipairs(input) do
      --    output[k] = self:recursiveMask(output[k], v, mask)
      -- end
      assert(false, 'Not available for table input')
   else
      assert(torch.isTensor(input))
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure mask has the same dimension as the input tensor
      local inputSize = input:size():fill(1)
      if self.batchmode then
         inputSize[1] = input:size(1)
      end
      mask:resize(inputSize)
      -- build mask
      local zeroMask = mask:expandAs(input)
      output:resizeAs(input):copy(input)

      -- if not first word in sequence
      if lastOutput:dim() ~=0 then
         output:maskedCopy(zeroMask, lastOutput)
      end
   end

   return output
end

function MaskZeroCopy:recursiveMaskBackward(output, input, mask)

   if torch.type(input) == 'table' then
      -- output = torch.type(output) == 'table' and output or {}
      -- for k,v in ipairs(input) do
      --    output[k] = self:recursiveMask(output[k], v, mask)
      -- end
      assert(false, 'Not available for table input')
   else
      assert(torch.isTensor(input))
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure mask has the same dimension as the input tensor
      local inputSize = input:size():fill(1)
      if self.batchmode then
         inputSize[1] = input:size(1)
      end
      mask:resize(inputSize)
      -- build mask
      local zeroMask = mask:expandAs(input)
      output:resizeAs(input):copy(input)
      output:maskedFill(zeroMask, 0)
   end

   return output
end

function MaskZeroCopy:updateOutput(input)
   -- recurrent module input is always the first one
   local rmi = self:recursiveGetFirst(input):contiguous()
   if rmi:dim() == self.nInputDim then
      self.batchmode = false
      rmi = rmi:view(-1) -- collapse dims
   elseif rmi:dim() - 1 == self.nInputDim then
      self.batchmode = true
      rmi = rmi:view(rmi:size(1), -1) -- collapse non-batch dims
   else
      error("nInputDim error: "..rmi:dim()..", "..self.nInputDim)
   end

   -- build mask
   local vectorDim = rmi:dim()
   self._zeroMask = self._zeroMask or rmi.new()
   self._zeroMask:norm(rmi, 2, vectorDim)
   self.zeroMask = self.zeroMask or (
       (torch.type(rmi) == 'torch.CudaTensor') and torch.CudaByteTensor()
       or (torch.type(rmi) == 'torch.ClTensor') and torch.ClTensor()
       or torch.ByteTensor()
    )
   self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)

   -- forward through decorated module
   local output = self.modules[1]:updateOutput(input)

   self.output = self:recursiveMask(self.output, output, self.zeroMask)
   return self.output
end

function MaskZeroCopy:updateGradInput(input, gradOutput)
   -- zero gradOutputs before backpropagating through decorated module
   if self.backward then
      self.gradOutput = self:recursiveMaskBackward(self.gradOutput, gradOutput, self.zeroMask)
   else
      self.gradOutput = self:recursiveMask(self.gradOutput, gradOutput, self.zeroMask)
   end

   self.gradInput = self.modules[1]:updateGradInput(input, self.gradOutput)
   return self.gradInput
end

function MaskZeroCopy:type(type, ...)
   self.zeroMask = nil
   self._zeroMask = nil
   self._maskbyte = nil
   self._maskindices = nil
   return parent.type(self, type, ...)
end