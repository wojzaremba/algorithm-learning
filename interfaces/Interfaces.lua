--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


local Interfaces = torch.class('Interfaces')

local unpack = unpack and unpack or table.unpack

function Interfaces:__init()
  self.data = Data()
  self.interfaces = {self.data}
  self.name2desc = {}
  for _, interface in pairs(self.interfaces) do
    for name, desc in pairs(interface.actions_type) do
      self.name2desc[name] = desc
    end
  end
  self.action_type = {}
  for k, v in pairs(self.data.actions_type) do
    self.action_type[k] = v
  end
  if self.tape then
    for k, v in pairs(self.tape.actions_type) do
      self.action_type[k] = v
    end
  end
  Interfaces.new_node = new_node
end

function Interfaces:apply(ins_node)
  for _, interface in pairs(self.interfaces) do
    ins_node = interface:apply(ins_node)
  end
  return ins_node
end

function Interfaces:set_sizes()
  params.input_size = 0
  for _, interface in pairs(self.interfaces) do
    for name, desc in pairs(interface.view_type) do
      params.input_size = params.input_size + (desc.size + 1)
    end
    for name, desc in pairs(interface.actions_type) do
      params.input_size = params.input_size + (desc.size + 1)
    end
  end
end

function Interfaces:q_learning(ins_node, h)
  local logits = nn.Linear(params.rnn_size, model.q_size)(nn.Tanh()(h))
  ins_node = self:q_logits(ins_node, logits)
  ins_node = self:q_sample(ins_node)
  return ins_node
end

function Interfaces:decode_q(chosen)
  local ret = {}
  for k, v in pairs(self.action_type) do
    ret[k] = (chosen - 1) % v.size + 1
    chosen = ((chosen - 1) - (ret[k] - 1)) / v.size + 1
  end
  return ret
end

function Interfaces:encode_q(vals)
  local ret = 0
  local order = {}
  local actions = self.action_type
  for k, v in pairs(actions) do
    table.insert(order, k)
  end
  for i = 1, #order do
    local k = order[#order - i + 1]
    local v = actions[k]
    ret = ret * v.size
    ret = ret + (vals[k] - 1)
  end
  return ret + 1
end

function Interfaces:q_logits(ins_node, logits)
  local function fp(self, input)
    local ins, logits = unpack(input)
    for i = 1, params.batch_size do
      local sample = ins.data.samples[i]
      ins.target_idx[i] = sample.target_idx
      ins.q[i]:copy(logits[i])
      ins.logits[i]:copy(logits[i])
    end
    return ins
  end
  local function bp(self, input, gradOutput)
    local ins, logits, logits_dropped = unpack(input)
    local dlogits = logits:clone():zero()
    for i = 1, params.batch_size do
      local sample = ins.data.samples[i]
      local mul = sample.normal - ins.target_idx[i]
      dlogits[i]:copy(ins.dq[i])
      if sample.train == 1 then
        assert(ins.dq[i]:norm() == 0)
      end
    end
    self.gradInput = {torch.zeros(1), dlogits}
    return self.gradInput
  end
  return self:new_node(ident, fp, bp, {ins_node, logits})
end

function Interfaces:q_sample(ins_node, logits)
  local function sample_actions(sample)
    assert(sample.train == 2)
    local acc_current = acc:get_current_acc(sample.complexity)
    local acc_ranges = {0, 0.9, 1}
    -- Probability of choosing a random action.
    local random = {20, 20}
    if params.decay_expr ~= nil then
      random = {20, -1}
    end
    for i = 1, #acc_ranges - 1 do
      if acc_current >= acc_ranges[i] and
         acc_current < acc_ranges[i + 1] and
         math.random(random[i]) == 1 then
        local a = {}
        for k, v in pairs(self.action_type) do
          a[k] = math.random(v.size)
        end
        return a
      end
    end
    return nil
  end

  local function fp(self, ins)
    local max_idx = ins.max_idx
    local chosen = ins.chosen
    ins:child().data.sampled:copy(ins.data.sampled)
    for i = 1, params.batch_size do
      local sample = ins.data.samples[i]
      max_idx[i] = argmax(ins.q[i])
      chosen[i] = max_idx[i]
      local a = self:decode_q(chosen[i])
      if sample.train == 2 then
        a = sample_actions(sample) or a
      end

      if self:encode_q(a) ~= max_idx[i] then
        ins.sampled[i] = 1
        if sample.sampled == nil then
          sample.sampled = ins.time
          ins:child().data.sampled[i] = 2
        end
      end
      chosen[i] = self:encode_q(a)
      assert(chosen[i] >= 1 and chosen[i] <= model.q_size)
      for k, v in pairs(self.action_type) do
        ins.actions[k].action[i] = a[k]
      end
    end
    return ins
  end
  return self:new_node(ident, fp, empty_bp, {ins_node})
end
