--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


function ident(x)
  return x
end

function empty()
  return {torch.zeros(1)}, {torch.zeros(1)}
end

function empty_bp(self, input, gradOutput, name)
  if name == nil then
    name = "gradInput"
  end
  if istable(input) then
    self[name] = {}
    for k, v in pairs(input) do
      self[name][k] = empty_bp(self, v, 0, k)
    end
  elseif type(input) == "number" or
         (type(input) == 'table' and input.__typename == "Instance") then
    if self[name] == nil then
      self[name] = torch.zeros(1)
    end
  elseif torch.isTensor(input) then
    if self[name] == nil then
      self[name] = input:clone():zero()
    end
  else
    print("input")
    print(input)
    print("gradOutput")
    print(gradOutput)
    assert(false)
  end
  assert(self[name] ~= nil)
  return self[name]
end

function empty_type(self, new_type)
  return self
end

function emb(self, ins_node, fun, selection, which)
  local function fp(name, which)
    local function forward(self, ins)
      return fun(ins, name, which)
    end
    return forward
  end
  assert(type(selection) == "table")
  local ret = {}
  for name, desc in pairs(selection) do
    if desc.size > 0 then
      local tmp = new_node(self, ident, fp(name, which), empty_bp, {ins_node})
      table.insert(ret, nn.OneHotIndex(desc.size, name)(tmp))
    end
  end
  return ret
end

function new_node(self, init, fp, bp, input_nodes, name, which)
 local module = safe_merge(table.copy(self), {updateOutput=fp,
                                              updateGradInput=bp,
                                              accGradParameters=empty,
                                              parameters=empty,
                                              type=empty_type})
  init(module)
  setmetatable(module, getmetatable(self))
  module.trace = debug.traceback()
  if name ~= nil then
    module.node_name = name
  end
  if which ~= nil then
    module.which = which
  end
  local node = nngraph.Node({module=module})
  if not istable(input_nodes) then
    input_nodes = {input_nodes}
  end
  for _, input_node in pairs(input_nodes) do
    if torch.typename(input_node) ~= 'nngraph.Node' then
      error('what is this in the input? ' .. tostring(input_node))
    end
    node:add(input_node, true)
  end
  return node
end

local unpack = unpack and unpack or table.unpack

local Module = torch.getmetatable('nn.Module')
local call__ = Module.__call__
function Module:__call__(...)
  local input = {...}
  local ret = call__(unpack(merge({self}, input)))
  if ret.data ~= nil and
     ret.data.module ~= nil then
    ret.data.module.trace = debug.traceback()
  end
  return ret
end
