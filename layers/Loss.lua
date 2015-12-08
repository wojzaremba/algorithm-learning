--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Computes cross entropy loss, and reward (it's stored in correct).
local Loss, parent = torch.class('nn.Loss', 'nn.Module')

local unpack = unpack and unpack or table.unpack

function Loss:__init()
  parent.__init(self)
  self.dlogprob = torch.Tensor()
  self.dtarget = zeros()
  self.dins = torch.zeros(1)
end

function Loss:updateOutput(input)
  local ins, logprob, target = unpack(input)
  ins:child().data.time:copy(ins.data.time):add(1)
  local begin = ins:child().data.begin
  local correct = ins.correct:zero()
  begin:copy(ins.data.begin)
  local finish = ins:child().data.finish:zero():add(1 / 0)
  local err = ins.err
  err:zero()
  local pred = ins:child().data.pred:zero()
  for i = 1, params.batch_size do
    pred[i] = params.vocab_size
    local sample = ins.data.samples[i]
    if not sample:eos() then
      for k = 1, logprob[i]:size(1) do
        if math.abs(logprob[i]:max() - logprob[i][k]) < 1e-8 then
          pred[i] = k
          break
        end
      end
      assert(pred[i] ~= 0)
      if target[i] ~= 0 then
        if sample:predict(pred[i], logprob[i]) then
          correct[i] = 1
        else
          correct[i] = 0
        end
        err[i] = -logprob[i][target[i]]
      else
        pred[i] = params.vocab_size
      end
      if model.step % 20 == 1 and params.train == 2 then
        assert(#sample.strings + ins.data.begin[i] == ins.time)
        -- Saves trace of an executions.
        table.insert(sample.strings, tostring(sample))
      end
    end
  end
  return ins
end

function Loss:updateGradInput(input, gradOutput)
  local ins, logprob, target = unpack(input)
  self.dlogprob:resizeAs(logprob):zero()
  for i = 1, target:size(1) do
    if ins.data.samples[i].train == 2 or params.gc == 1 then
      if target[i] ~= 0 then
        self.dlogprob[i][target[i]] = -1
      end
      model.trained_chars = model.trained_chars + 1
    end
  end
  self.gradInput = {self.dins, self.dlogprob, self.dtarget}
  return self.gradInput
end
