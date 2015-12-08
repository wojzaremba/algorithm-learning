--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- An abstract Interface class. Every interface (i.e. tape, grid, memory)
-- has to inherit from this class.
local Interface = torch.class('Interface')

function Interface:__init(view_type, actions_type)
  self.view_type = view_type
  self.actions_type = actions_type
  self.name = self.__typename:lower()
  Interface.emb = emb
  Interface.new_node = new_node
end

-- Returns the last action from the given Interface.
function Interface:last_action(ins_node)
  local function action(ins, name)
    local ret = nil
    local delay = 1
    if ins.time - delay < 1 then
      ret = ins.actions[name].action:clone()
      ret:zero():add(self.actions_type[name].size + 1)
    else
      local past = model(ins.time - delay)
      ret = past.actions[name].action:clone()
      for i = 1, params.batch_size do
        if past.data.samples[i] ~= ins.data.samples[i] then
          ret[i] = self.actions_type[name].size + 1
        end
      end
    end
    return ret 
  end 
  local last_action = emb(self, ins_node, action, self.actions_type)
  return last_action
end

function Interface:__tostring__()
  return torch.type(self)
end
