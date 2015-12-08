--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Defines the model, its forward and backward pass. Contains model parameters.
local Model = torch.class('Model')

function Model:__init()
  self.step = 1
  self.trained_chars = 0
  self.instances = {}
  self.fields = {}
  self.samples = {}
  self.norm_dw = 0
  self.q_size = 1
  self.ring = 5
  for _, s in pairs(interfaces.name2desc) do
    self.q_size = self.q_size * s.size
  end
  self:clean()
end

function Model:clean()
  local f = self.fields
  if self.fields.s ~= nil then
    local function zero(obj) 
      for k, _ in pairs(obj) do
        if torch.isTensor(obj[k]) then
          obj[k]:zero()
        elseif type(obj[k]) == "table" then
          zero(obj[k])
        end
      end
    end
    zero(self.fields)
    for name, desc in pairs(interfaces.name2desc) do
      if desc.size > 0 then
        f.actions[name].action:zero():add(desc.size)
      end
    end
    return
  end
  local tl = params.max_seq_length + 2
  local bs = params.batch_size
  local rs = params.rnn_size
  local ks = params.key_size
  local ms = params.memory_size
  local quant = params.layers
  -- LSTM has twice as many units (half for hidden states, half for cells).
  if params.unit == "lstm" then
    quant = quant * 2
  end
  f.s = torch.zeros(tl, quant, bs, rs) -- State.
  f.ds = f.s:clone() -- Derivative of state.
  f.correct = torch.zeros(tl, bs) -- If prediction is correct.
  f.dq = torch.zeros(tl, bs, self.q_size) -- Derivative with respect to q-function.
  f.err = torch.zeros(tl, bs) -- Error.
  f.q = torch.zeros(tl, bs, self.q_size) -- Q-value.
  f.logits = torch.zeros(tl, bs, self.q_size)
  f.chosen = torch.zeros(tl, bs) -- Which action have been choosen
  f.sampled = torch.zeros(tl, bs) -- If action was sampled.
  f.target_idx = torch.zeros(tl, bs)
  f.max_idx = torch.zeros(tl, bs) -- Which Q-function index yields the highest value.
  f.data = {} 
  f.tape = {}
  f.total = {}
  f.actions = {}
  f.d = {}
  self:register_empty("data", {"begin", "finish", "time", "sampled", "x", "y", "pred", "task"})

  f.actions = {}
  for name, desc in pairs(interfaces.name2desc) do
    f.actions[name] = {}
    f.actions[name] = {}
    f.actions[name].action = torch.zeros(tl, bs):add(desc.size)      
    f.actions[name].max = torch.zeros(tl, bs)
  end
  f.tape.idx = torch.zeros(tl, bs)
end

function Model:register_empty(name, fields)
  if self.fields[name] == nil then
    self.fields[name] = {}
  end
  for _, f in pairs(fields) do
    self.fields[name][f] = torch.zeros(params.max_seq_length + 2, params.batch_size)
  end
end

-- model(t) gives an access to the model instantiation ot time t.
function Model:__call(time)
  assert(type(time) == "number")
  local offset = 5
  if params.train == 2 or time < offset then
    if model.instances[time] == nil then
      model.instances[time] = Instance(time)
    end
   return  model.instances[time]
  else
    -- While testing on the very long sequences, we have to
    -- reuse previous time instances (to not run out of memory).
    for i = -1, 1 do
      local idx = (time - offset + i) % self.ring + offset
      if model.instances[idx] == nil then
        model.instances[idx] = Instance(idx)
      end
      model.instances[idx].time = time + i 
    end
    return model.instances[(time - offset)  % self.ring + offset]
  end
end

function Model:reboot()
  g_make_deterministic(params.seed)
  self.root = model(0)
  paramdx:zero()
  for _, interface in pairs(interfaces.interfaces) do
    interface:clean()
  end
  interfaces.data:clean()
  model:clean()
  acc:clean()
end

-- Computes rewards, and their derivatives.
function Model:rewards()
  -- If during test, then exit (params.train \in {1, 2}).
  if params.train == 1 then
    return
  end
  local err = 0
  for batch = 1, params.batch_size do
    for i = 1, params.seq_length - 1 do
      local ins = model(i)
      local sample = ins.data.samples[batch]
      if sample:eos() and sample.train == 2 then
        local diff = _G[params.q_type](ins, batch)
        assert(ins.dq[batch]:norm() == 0)
        if acc:get_current_acc(sample.complexity) > 0.9 then
          local q_decay_lr = params.q_decay_lr or 0
          ins.dq[batch]:add(q_decay_lr * (ins.q[batch]:sum() - 1))
          for j = 1, ins.dq:size(2) do
            local q = ins.q[batch][j]
            if q <= 0 then
              ins.dq[batch][j] = ins.dq[batch][j] + q_decay_lr * q 
            elseif q >= 1 then
              ins.dq[batch][j] = ins.dq[batch][j] - q_decay_lr * (q - 1)
            end
          end
        end
        local chosen = ins.chosen[batch]
        ins.dq[batch][chosen] = ins.dq[batch][chosen] + params.q_lr * diff
        err = err + params.q_lr * (diff * diff) / 2
      end
    end
  end
  model.err = model.err + err / params.batch_size
end

-- Forward propagation.
function Model:fp()
  -- Sequence length to unroll is chosen dynamically depending on the current complexity.
  params.seq_length = 1
  model:clean()
  interfaces.data:clean()
  model.err = 0 
  local time_offset = 1
  while true do
    local ins = model(time_offset)
    local s = ins.rnn:forward({ins, ins.s})[2]
    local sample = ins.data.samples[1]
    tensors_copy(model(time_offset + 1).s, s)
    model.err = model.err + ins.err:mean()
    if params.train == 1 then
      io.write(tostring(time_offset) .. " ")
      io.flush()
    end
    time_offset = time_offset + 1
    if (params.train == 1 and time_offset > params.test_len) or
       (params.train == 2 and time_offset > params.seq_length) then
      break
    end
  end
  self:rewards()
  visualizer:visualize()
  self.samples = {}
end

-- Backpropagation.
function Model:bp()
  -- Don't backpropagate with respect to the test data.
  if params.train == 1 then
    return
  end
  paramdx:zero()
  for time_offset = params.seq_length, 1, -1 do
    local ins = model(time_offset)
    assert(ins.s ~= nil and ins.ds ~= nil)
    local ds = ins.rnn:backward({ins, ins.s}, {torch.zeros(1), ins.ds})[2]
    tensors_copy(model(time_offset - 1).ds, ds)
  end
  paramdx:div(params.batch_size * params.seq_length / 10)
  collectgarbage()
end

