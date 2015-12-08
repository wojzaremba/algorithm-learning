--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Moves over input tape, or input grid.
local Move, parent = torch.class('nn.Move', 'nn.Module')

function Move:__init()
  parent.__init(self)
  self.dins = torch.zeros(1)
end

function Move:updateOutput(input)
  local ins = input
  local err = ins.err
  local time = ins:child().data.time
  local pred = ins:child().data.pred
  local sampled = ins:child().data.sampled
  local x = ins:child().data.x:zero():add(1)
  local y = ins:child().data.y:zero():add(1)
  local samples = {}
  for i = 1, params.batch_size do 
    local sample = ins.data.samples[i]
    samples[i] = sample
    sample:move()
    if sample:eos() then
      acc:record(sample)
      time[i] = 1
      if params.train == 2 then
        ins:child().data.begin[i] = ins:child().time
        for j = math.max(ins.data.begin[i], 1), ins.time do
          assert(model(j).data.finish[i] == 1 / 0)
          model(j).data.finish[i] = ins.time
        end
      end
      local sample = curriculum:generateNewSample()
      table.insert(model.samples, sample)
      sample.ins = ins:child().time
      sample.idx = i
      pred[i] = params.vocab_size
      samples[i] = sample
      sampled[i] = 1
    end
    x[i] = samples[i]:current_input() 
    y[i] = samples[i]:current_target()
  end
  ins:child().data.samples = samples
  return ins
end

function Move:updateGradInput(input, gradOutput)
  self.gradInput = self.dins
  return self.gradInput
end
