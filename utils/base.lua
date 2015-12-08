--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


function g_make_deterministic(seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
end

function join_table(table)
  if #table > 1 then
    return nn.JoinTable(2)(table)
  else
    return table[1]
  end
end

function clone(from, val)
  if torch.isTensor(from) then
    local ret = from:clone()
    if val ~= nil then
      ret:zero():add(val)
    end
    return ret
  elseif type(from) == "number" then
    if val ~= nil then
      return val
    else
      return from
    end
  elseif type(from) == "string" or
         type(from) == "boolean" then
    return from
  end
  assert(type(from) == "table")
  local ret = {}
  for k, v in pairs(from) do 
    ret[k] = clone(v, val)
  end
  return ret
end

-- Display float up to 5 digits after dot.
function g_f5(f)
  local ret = string.format("%.5f", f)
  if #tostring(f) < #ret then
    return tostring(f)
  else
    return ret
  end
end

-- Display float up to 3 digits after dot.
function g_f3(f)
  local ret = string.format("%.3f", f)
  if #tostring(f) < #ret then
    return tostring(f)
  else
    return ret
  end
end

-- Display int.
function g_d(f)
  return string.format("%d", math.floor(f))
end

function istable(x)
  return type(x) == 'table' and not torch.typename(x)
end

function merge(a, b)
  local ret = setmetatable({}, getmetatable(a))
  for k, v in pairs(a) do
    if type(k) == "number" then
      ret[#ret + 1] = v
    else
      ret[k] = v
    end
  end
  if b ~= nil then
    for k, v in pairs(b) do
      if type(k) == "number" then
        ret[#ret + 1] = v
      else
        ret[k] = v
      end
    end
  end
  return ret
end

function safe_merge(a, b)
  local ret = {}
  for k, v in pairs(a) do
    ret[k] = v
  end
  if b ~= nil then
    for k, v in pairs(b) do
      assert(ret[k] == nil)
      ret[k] = v
    end
  end
  return ret
end

function table.copy(obj)
  local ret = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do
    ret[k] = v
  end
  return ret
end

function tensors_copy(to, from)
  if torch.isTensor(from) then
    to:copy(from)
  else
    for k, v in pairs(from) do
      tensors_copy(to[k], v)
    end
  end
end

function one()
  return torch.ones(1)
end

function zeros(size)
  if size ~= nil then
    return torch.zeros(params.batch_size, size)
  else
    return torch.zeros(params.batch_size)
  end
end

function ones(size)
  if size ~= nil then
    return torch.ones(params.batch_size, size)
  else
    return torch.ones(params.batch_size)
  end
end

function split(node, nr)
  assert(nr >= 1)
  if nr == 1 then
    return {nn.SelectTable(1)(node)}
  else
    return {node:split(nr)}
  end
end

function len(T)
  if type(T) == "string" then
    local inside = false
    local count = 0 
    for i = 1, #T - 1 do
      if not inside and T:byte(i, i) == 27 and T:sub(i + 1, i + 1) == "[" then
        inside = true
      end
      if not inside then
        count = count + 1
      end
      if inside and T:sub(i, i) == "m" then
        inside = false
      end
    end
    if not inside then
      count = count + 1
    end
    return count
  else
    local count = 0
    for _ in pairs(T) do 
      count = count + 1 
    end
    return count
  end
end

function argmax(x)
  local max = -1 / 0
  local ret = 0
  for j = 1, x:size(1) do
    if max < x[j] then
      max = x[j]
      ret = j
    end
  end
  return ret
end

