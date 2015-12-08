--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Encodes integers in dummy vector representation.
-- i.e. 4 would be encoded as [0, 0, 0, 1, 0, 0 ...]
local Index, parent = torch.class('nn.OneHotIndex', 'nn.Module')

function Index:__init(inputSize, name)
  parent.__init(self)
  assert(type(inputSize) == "number")
  self.inputSize = inputSize + 1
  self.name = name
end

function Index:updateOutput(input)
  self.output:resize(input:size(1), self.inputSize):zero()
  for i = 1, input:size(1) do
    assert(input[i] >= 1 and input[i] <= self.inputSize)
    self.output[i][input[i]] = 1
  end
  return self.output
end

function Index:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resize(input:size())
    return self.gradInput
  end
end

Index.sharedAccUpdateGradParameters = Index.accUpdateGradParameters
