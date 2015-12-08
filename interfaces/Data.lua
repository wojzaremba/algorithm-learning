--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Interface to access input tape (or grid), and output tape.
local Data, parent = torch.class('Data', 'Interface')

function Data:__init()
  assert(type(params.ntasks) == "number")
  print(params)
  parent.__init(self, {x           = {size=params.vocab_size}, --
                       pred        = {size=params.vocab_size}, -- this are inputs to the interface
                       sampled     = {size=2}},                -- 
                      {xa          = {size=math.pow(2, params.dim), now=1}, -- where to move.
                       ya          = {size=2, now=2}}) -- do prediction or not

end


function Data:view(ins_node)
  local function ident(ins, name)
    local ret = nil
    local delay = 0 
    if ins.time - delay < 1 then
      ret = ins.data[name]:clone()
      ret:zero():add(self.view_type[name].size)
    else
      local past = model(ins.time - delay)
      ret = past.data[name]:clone()
      if delay > 0 then
        for i = 1, params.batch_size do
          if past.data.samples[i] ~= ins.data.samples[i] then
            ret[i] = self.view_type[name].size
          end
        end
      end
    end
    return ret 
  end
  return self:emb(ins_node, ident, self.view_type)
end

function Data:apply(ins_node)
  local function fp(self, ins)
    local xa = ins.actions.xa.action
    for i = 1, params.batch_size do      
      local sample = ins.data.samples[i]
      sample.dir = Game.dirs[sample.dim][xa[i]]
    end
    return ins
  end
  return self:new_node(ident, fp, empty_bp, {ins_node}) 
end

function Data:clean()
  local child = model.root:child()
  child.data.begin:fill(1)
  child.data.finish:fill(1):mul(1 / 0)
  child.data.time:fill(1)
  child.data.pred:fill(params.vocab_size)
  child.data.x:fill(1)
  child.data.y:fill(1)
  child.data.sampled:fill(1)
  child.data.samples = {}
  for i = 1, params.batch_size do
    local sample = curriculum:generateNewSample()
    sample.ins = child.time
    sample.idx = i
    child.data.x[i] = sample:current_input()
    child.data.y[i] = sample:current_target()
    child.data.samples[i] = sample
    table.insert(model.samples, sample)
  end 
  collectgarbage()
end

function Data:target(ins_node)
  local function fp(self, ins)
    self.output = zeros() 
    for idx = 1, params.batch_size do
      local sample = ins.data.samples[idx]
      if ins.actions.ya.action[idx] == 2 then
        local sample = ins.data.samples[idx]
        self.output[idx] = ins.data.y[idx]
      end
    end
    return self.output
  end
  local target = self:new_node(ident, fp, empty_bp, {ins_node}) 
  return target
end

