--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


function gru_core(x, prev_h)
  local function new_input_sum(xv, hv)
    local i2h = nn.Linear(params.rnn_size, params.rnn_size)(xv)
    local h2h = nn.Linear(params.rnn_size, params.rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end
  local update_gate = nn.Sigmoid()(new_input_sum(x, prev_h))
  local reset_gate = nn.Sigmoid()(new_input_sum(x, prev_h))
  -- compute candidate hidden state
  local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
  local p2 = nn.Linear(params.rnn_size, params.rnn_size)(gated_hidden)
  local p1 = nn.Linear(params.rnn_size, params.rnn_size)(x)
  local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
  -- compute new interpolated hidden state, based on the update gate
  local zh = nn.CMulTable()({update_gate, hidden_candidate})
  local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
  local next_h = nn.CAddTable()({zh, zhm1})
  return next_h
end

function gru(single_input, prev_s)
  local i                = {[0] = single_input}
  local next_s           = {}
  local split = split(prev_s, params.layers)
  for layer_idx = 1, params.layers do
    local prev_h         = split[layer_idx]
    local next_h = gru_core(i[layer_idx - 1], prev_h)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  return i[params.layers], nn.Identity()(next_s)
end

function feedforward(h, prev_s)
  return h, prev_s
end

function lstm_core(x, prev_c, prev_h)
  local i2h              = nn.Linear(params.rnn_size, 4 * params.rnn_size)(x)
  local h2h              = nn.Linear(params.rnn_size, 4 * params.rnn_size)(prev_h)
  local gates            = nn.CAddTable()({i2h, h2h})
  local reshaped_gates   = nn.Reshape(4, params.rnn_size)(gates)
  local sliced_gates     = nn.SplitTable(2)(reshaped_gates)
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


function lstm(single_input, prev_s)
  local i                = {[0] = single_input}
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local next_c, next_h
    next_c, next_h = lstm_core(i[layer_idx - 1], prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  return i[params.layers], nn.Identity()(next_s)
end


