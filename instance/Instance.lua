--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- It contains all the information about a single time-step (i.e. gradients
-- of the RNN, actions taken etc.).
local Ins = torch.class('Instance')

function Ins:__init(time)
  self.time = time
  assert(type(self.time) == "number")
  self:copy_from_root()
  local function assign(from, to) 
    for k, v in pairs(from) do
      if torch.isTensor(v) then
        to[k] = v[self.time + 1]
      elseif type(v) == "table" then
        to[k] = {}
        assign(v, to[k])
      end
    end
  end
  assign(model.fields, self)
end

-- Return previous instance in time.
function Ins:prev()
  return model(self.time - 1)
end

function Ins:copy_from_root()
  if model.core_network == nil then
    model.core_network = create_network()
    -- Network is created only once. Other instances contain reference 
    -- to the same weights. 
    paramx, paramdx = model.core_network:getParameters()
    paramx:mul(2)
  end
  self.rnn = create_network()
  for i, node in pairs(self.rnn.forwardnodes) do
    if node.data.module then
      local to = node.data.module
      local from = model.core_network.forwardnodes[i].data.module
      if to.weight then
        to.weight = from.weight
        to.gradWeight = from.gradWeight
        to.bias = from.bias
        to.gradBias = from.gradBias
      end
    end
  end
end

-- Return next instance in time.
function Ins:child()
  return model(self.time + 1)
end



