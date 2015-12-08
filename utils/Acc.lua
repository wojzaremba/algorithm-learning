--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


local Acc = torch.class('Acc')

function Acc:__init()
  self.fields = {"correct", "normal"}
  self.min_normal = {10, 50}
  self.alpha = 0.9
  self.min_acc = 0.995
  self:clean()
end

function Acc:getkey(current_complexity, complexity)
  assert(current_complexity ~= nil and complexity ~= nil)
  local ret = "current_complexity=" .. tostring(current_complexity)
  ret = ret .. string.format(", complexity:%03d", complexity)
  self.complexity[ret] = complexity
  if self.available_complexity[current_complexity] == nil then
    self.available_complexity[current_complexity] = {}
  end
  self.available_complexity[current_complexity][complexity] = true
  self.current_complexity[ret] = current_complexity
  return ret
end

function Acc:is_normal()
  local count = 0 
  if self.available_complexity == nil or 
     self.available_complexity[curriculum.complexity] == nil then
    return false
  end
  for c, _ in pairs(self.available_complexity[curriculum.complexity]) do
    if (params.train == 1 and c > curriculum.complexity + 10) or (c >= curriculum.complexity and c <= curriculum.complexity + 10 and params.train == 2) then
      local key_c = self:getkey(curriculum.complexity, c)
      count = count + 1
      if self.strict_normal[key_c] < self.min_normal[params.train] then
        return false
      end
    end
  end
  if count == 0 then
    return false
  end
  return true
end

function Acc:is_good(min_acc)
  min_acc = min_acc or self.min_acc
  local count = 0 
  if self.available_complexity == nil or 
     self.available_complexity[curriculum.complexity] == nil then
    return false
  end
  for c, _ in pairs(self.available_complexity[curriculum.complexity]) do
    if (params.train == 1 and c > curriculum.complexity + 10) or (c >= curriculum.complexity and c < curriculum.complexity + 10 and params.train == 2) then
      local key_c = self:getkey(curriculum.complexity, c)
      count = count + 1
      if self.strict_acc[key_c] < min_acc then
        return false
      end
    end
  end
  if count == 0 then
    return false
  end
  return true
end

function Acc:record(sample)
  if not sample:eos() or 
     sample.train == 2 then
    return
  end
  local key = self:getkey(curriculum.complexity, sample.complexity)
  for _, f in pairs(self.fields) do
    if self[f][key] == nil then
      self[f][key] = sample[f]
    end
    self[f][key] = self[f][key] + sample[f]
  end
  if self.strict_normal[key] == nil then
    self.strict_normal[key] = 0
    self.strict_correct[key] = 0
  end
  if sample.correct == sample.normal then 
    self.strict_correct[key] = self.strict_correct[key] + 1
  elseif sample.normal > 20 and sample.correct == sample.normal - 1 then
    self.strict_correct[key] = self.strict_correct[key] + 1
    self.correct[key] = self.correct[key] + 1
  end

  self.strict_normal[key] = self.strict_normal[key] + 1
  self.strict_acc[key] = self.strict_correct[key] / self.strict_normal[key] 
  self.acc[key] = self.correct[key] / self.normal[key] 

  while self.normal[key] > 100 and self.alpha ~= 1 do
    self.normal[key] = self.normal[key] * self.alpha
    self.correct[key] = self.correct[key] * self.alpha
  end
  while self.strict_normal[key] > 100 and self.alpha ~= 1 do
    self.strict_normal[key] = self.strict_normal[key] * self.alpha
    self.strict_correct[key] = self.strict_correct[key] * self.alpha
  end
end

function Acc:clean()
  for _, f in pairs(self.fields) do
    self[f] = {}
  end
  self.complexity = {}
  self.available_complexity = {}
  self.current_complexity = {}
  self.acc = {}
  self.strict_acc = {}
  self.strict_correct = {}
  self.strict_normal = {}
end

function Acc:__tostring__()
  local function get_res(field)
    local current_complexities = {}
    for k, v in pairs(self.current_complexity) do 
      if current_complexities[v] == nil then
        current_complexities[v] = {}
      end
      table.insert(current_complexities[v], k)
      table.sort(current_complexities[v])
    end
    local results = "{"
    for current_complexity, list in pairs(current_complexities) do
      results = results .. tostring(current_complexity) .. " : ["
      for _, key in pairs(list) do
        results = results .. "(" .. g_d(self.complexity[key]) .. ", " .. g_f3(self[field][key]) .. "), "
      end
      results = results .. "], "
    end
    return results .. "#\n"
  end
  return "\nAccuracies for a given complexity.\n" .. 
    "Accuracy on the whole sample basis = " .. get_res("strict_acc") .. 
    "Accuracy on the character basis = " .. get_res("acc")
end

function Acc:get_current_acc(complexity)
  complexity = complexity or curriculum.complexity
  for key, _ in pairs(self.complexity) do
    if self.complexity[key] == complexity and
       self.current_complexity[key] == curriculum.complexity then
      return self.acc[key]
    end
  end
  return 0
end



