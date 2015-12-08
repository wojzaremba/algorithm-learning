--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Cleans state between samples (i.e. when the episode is over).
-- It makes sure that nothing leaks during forward, as well as the backward pass.
local EpisodeBarrier, parent = torch.class('nn.EpisodeBarrier', 'nn.Module')

function EpisodeBarrier:__init(current)
  self.current = current
  self.buffers = {}
end

function EpisodeBarrier:clone_with_barrier(input, time, idx)
  if torch.isTensor(input) then
    if input:nDimension() == 2 then
      if self.buffers[idx] == nil then
        self.buffers[idx] = input:clone()
      end
      self.buffers[idx]:copy(input)
      local output = self.buffers[idx]
      idx = idx + 1
      assert(output:size(1) == time:size(1))
      assert(output:size(1) == params.batch_size)
      for i = 1, params.batch_size do
        if time[i] == 1 then
          output[i]:zero()
        end
      end
      return output
    else
      local ret = {}
      for k = 1, input:size(1) do
        idx = idx + 1
        ret[k] = self:clone_with_barrier(input[k], time, idx)
      end
      return ret
    end
  elseif type(input) == "table" then
    local ret = {}
    for k, v in pairs(input) do
      idx = idx + 1
      ret[k] = self:clone_with_barrier(v, time, idx)
    end
    return ret
  else
    assert(false)
  end
end

local unpack = unpack and unpack or table.unpack

function EpisodeBarrier:updateOutput(input)
  local ins, input = unpack(input)
  return self:clone_with_barrier(input, ins.data.time, 1)
end

function EpisodeBarrier:updateGradInput(input, gradOutput)
  local ins, s = unpack(input)
  self.gradInput = {torch.zeros(1), self:clone_with_barrier(gradOutput, ins.data.time, 100)}
  return self.gradInput
end
