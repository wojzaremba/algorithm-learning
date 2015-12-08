--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


function q_watkins_dynamic(ins, batch)
  local last = 0 
  local diff = 0 
  local rewards = 0 
  local normal = ins.data.samples[batch].normal 
  local my_v = normal - ins.target_idx[batch] + 1
  for k = ins.time, ins.data.finish[batch] do
    if k == ins.time or model(k).sampled[batch] == 0 then
      rewards = rewards + model(k).correct[batch]
    else
      last = k
      break
    end
  end
  local sample = ins.data.samples[batch]
  local chosen = ins.chosen[batch]
  diff = ins.q[batch][chosen] 
  assert(my_v > 0)
  if last ~= 0 then
    assert(sample == model(last).data.samples[batch])
    local max_idx = model(last).max_idx[batch]
    local last_v = normal - model(last).target_idx[batch] + 1
    assert(last_v <= my_v)
    diff = diff - rewards / my_v - last_v * model(last).q[batch][max_idx] / my_v
  else
    diff = ins.q[batch][chosen] - (rewards / my_v)
  end
  return diff
end

function q_watkins(ins, batch)
  local discount = params.q_discount
  if discount == -1 then
    return q_watkins_dynamic(ins, batch)
  end
  local last = 0 
  local diff = 0 
  local rewards = 0 
  local mul = 1
  for k = ins.time, ins.data.finish[batch] do
    if k == ins.time or model(k).sampled[batch] == 0 then
      rewards = rewards + model(k).correct[batch] * mul
    else
      last = k
      break
    end
    mul = mul * discount
  end
  local sample = ins.data.samples[batch]
  local chosen = ins.chosen[batch]
  diff = ins.q[batch][chosen] - rewards
  if last ~= 0 then
    assert(sample == model(last).data.samples[batch])
    local max_idx = model(last).max_idx[batch]
    diff = diff - mul * model(last).q[batch][max_idx]
  end
  return diff
end

function q_classic_dynamic(ins, batch)
  local normal = ins.data.samples[batch].normal 
  local my_v = normal - ins.target_idx[batch] + 1
  local chosen = ins.chosen[batch]
  local diff = ins.q[batch][chosen] - ins.correct[batch] / my_v
  if ins:child().data.samples[batch] == ins.data.samples[batch] then
    local last_v = normal - ins:child().target_idx[batch] + 1
    assert(last_v <= my_v)
    local max_idx = ins:child().max_idx[batch]
    local discount = last_v / my_v
    diff = diff - discount * ins:child().q[batch][max_idx]
  end
  return diff
end

function q_classic(ins, batch)
  local discount = params.q_discount
  if discount == -1 then
    return q_classic_dynamic(ins, batch)
  end
  local chosen = ins.chosen[batch]
  local diff = ins.q[batch][chosen] - ins.correct[batch]
  if ins:child().data.samples[batch] == ins.data.samples[batch] then
    local max_idx = ins:child().max_idx[batch]
    diff = diff - discount * ins:child().q[batch][max_idx]
  end
  return diff
end

