--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Defines tasks.
local DataGenerator = torch.class('DataGenerator')

function DataGenerator:__init()
  self.task_ids = {}
  local tasks = 0
  self.base = 10

  for k, _ in pairs(getmetatable(self)) do
    if k:sub(1, 4) == "task" then
      tasks = tasks + 1
      self.task_ids[k:sub(6, #k)] = tasks
    end
  end
  params.ntasks = len(self.task_ids)
end

function DataGenerator:encode(s)
  local ret = ""
  if type(s) == "number" then
    ret = encode_map[s]
  else
    ret = ""
    for i = 1, s:size(1) do
      ret = ret .. encode_map[s[i]]
    end
  end
  assert(ret ~= nil, tostring(s))
  return ret
end

---------------------------------------------------

local function simple()
  local current = curriculum.complexity
  if current == 6 then
    return math.min(math.random(2), self.base - 1)
  elseif current == 10 then
    return math.min(math.random(3), self.base - 1)
  elseif current == 14 then
    return math.random(self.base) - 1
  end
end

-- Single digit multiplication.
function DataGenerator:task_single_mul(complexity)
  local inp = {}
  local len = math.max(complexity, 1)
  local digit = math.min(math.random(8) + 1, self.base - 1)
  for j = 1, len do
    inp[len - j + 1] = math.random(self.base) - 1
  end
  local ret = self:mul(digit, inp)
  if #ret.targets < 1 then
    return self:task_single_mul(complexity)
  end
  return ret
end

function DataGenerator:mul(digit, inp)
  local function at(self, x, y)
    if x == 1 and y == 1 then
      return digit 
    end
    if y == 2 and x >= -#inp + 2 and x <= 1 then
      return inp[x + #inp - 1]
    end
    if x == 3 then
      return "q"
    end
    return 0
  end
  local task = {{}}
  local config = {complexity=complexity,
                  dim=2,
                  at=at,
                  evaluate=function() end}
  local ret = Game(task, config)

  local curry_val = 0
  for i = 1, #inp + 1 do
    local x = inp[#inp - i + 1]
    if x == nil or x == self.base then
      x = 0
    end
    local out = x * digit + curry_val
    curry_val = math.floor(out / self.base)
    local out = out % self.base
    table.insert(ret.targets, out)
  end
  ret.complexity = #inp + 2
  ret.normal = #ret.targets
  return ret
end

-- Generic addition (arbitrary number of numbers).
function DataGenerator:add(inp)
  generic = generic or 0
  assert(inp ~= nil)
  local function at(self, x, y)
    for k = 1, #inp do
      if y == k and x >= -(#inp[k]) + 2 and x <= 1 then
        return inp[k][x + (#inp[k]) - 1]
      end
    end
    if x == 3 then
      return "q"
    end
    return "e"
  end
  local task = {{}}
  local config = {complexity=complexity,
                  dim=2,
                  at=at,
                  evaluate=function() end}
  local ret = Game(task, config)
  local curry_val = 0
  local max_len = 0
  for i = 1, #inp do
    max_len = math.max(max_len, #inp[i])
  end
  for i = 1, max_len + 1 do
    local out = 0
    for k = 1, #inp do
      local x = inp[k][#inp[k] - i + 1]
      if x == nil or x == self.base then
        x = 0
      end
      out = out + x
    end
    out = out + curry_val
    curry_val = math.floor(out / self.base)
    table.insert(ret.targets, out % self.base)
  end
  ret.len = (max_len + 1) * #inp 
  ret.normal = #ret.targets
  return ret
end

-- Two row addition.
function DataGenerator:task_addition(complexity)
  local inp = {}
  local rows = 2
  local min = math.random(3)
  local max_len = 0
  for i = 1, rows do
    local len = math.max(math.floor(complexity / rows), min)
    inp[i] = {}
    for j = 1, len do
      inp[i][len - j + 1] = math.random(self.base) - 1
    end
    max_len = math.max(max_len, len)
  end
  local ret = self:add(inp, generic)
  ret.complexity = rows * max_len
  if #ret.targets < 1 then
    return self:task_addition(complexity, rows, generic)
  end
  return ret
end

-- Three row addition.
-- We had to provide heavy curriculum to be able to solve this task.
function DataGenerator:task_addition3(complexity)
  local inp = {}
  local rows = 3
  local len = {}
  local max_len = 0
  local min_rows = math.random(3) + 1
  local acc_current = acc:get_current_acc(complexity)
  local curr_complexity = curriculum.complexity
  for i = 1, rows do
    len[i] = math.max(math.floor(complexity / rows), min_rows)
    inp[i] = {}
    for j = 1, len[i] do
      if params.train == 2 and (curr_complexity <= 4 or (complexity <= 16 and math.random(2) == 1)) then
        if acc_current < 0.7 then
          inp[i][j] = math.random(self.base - 1)
          if math.random(2) == 1 then
            inp[i][j] = math.min(math.random(4), self.base - 1)
          end
        else
          inp[i][j] = math.random(self.base - 1)
        end
      else
        inp[i][j] = math.random(self.base) - 1
      end
    end
    max_len = math.max(max_len, len[i])
  end
  local function simple(harder)
    if curr_complexity <= 4 and hardner ~= true then
      if math.random(3) <= 2 then
        return 0
      else
        return 1
      end
    elseif curr_complexity <= 8 then
      return math.min(math.random(4), self.base) - 1
    else
      if math.random(3) == 1 then
        return 0 
      elseif math.random(2) == 1 then
        return math.min(math.random(4), self.base) - 1
      else
        return math.random(self.base) - 1
      end
    end
  end
  local function set_simple(row, offset, harder)
    if #inp[row] > offset then
      inp[row][#inp[row] - offset] = simple(harder)
    end
  end
  if curr_complexity <= 12 or (curr_complexity <= 16 and math.random(2) == 1) then 
    if math.random(3) <= 2 then
      set_simple(1, 0) 
      set_simple(2, 0)
    else 
      set_simple(math.random(2), 0)
    end
    local acc_current12 = acc:get_current_acc(12)
    if max_len >= 2 then
      set_simple(3, 1)
      set_simple(2, 1, true)
    end
    if max_len >= 3 then
      set_simple(1, 2)
      set_simple(2, 2, true)
    end
    if max_len >= 4 then
      set_simple(3, 3)
      set_simple(2, 3, true)
    end
  end
  local ret = self:add(inp)
  ret.complexity = #inp * 3
  if #ret.targets < 1 then
    return self:task_addition(complexity, rows, generic)
  end
  ret.inp = inp
  return ret
end

-- Copy task.
function DataGenerator:task_copy(complexity)
  local tabs = {}
  tabs[complexity + 1] = {"q"}
  local config = {complexity=complexity,
                  dim=1}
  local ret = Game(tabs, config)
  ret.complexity = complexity
  return ret
end

-- Reverse task.
function DataGenerator:task_reverse(complexity)
  local function fun_p(self)
    self.output = true
    self.dir[1] = -1
    self.dir[2] = 0 
    self.dir_idx = 1 
  end 
  local function fun_init(self)
    self.output = false
  end 
  local time = math.floor(complexity / 2) + 1
  local task = {}
  assert(time >= 2)
  for i = 1, time - 1 do
    task[i] = {math.random(self.base) - 1}
  end
  task[time] = {"p"}
  task[time + 1] = {"p"}
  task[0] = {"q"}
  local config = {complexity=complexity,
                  dim=1,
                  p_loc=time,
                  init=fun_init,
                  fun_p=fun_p}
  return Game(task, config)
end

-- Walk task.
function DataGenerator:task_walk(complexity)
  local function fun_l(self)
    self.output = true;self.dir[1] = -1;self.dir[2] = 0;self.dir_idx = 1 
  end 
  local function fun_u(self)
    self.output = true;self.dir[1] = 0;self.dir[2] = -1;self.dir_idx = 4
  end 
  local function fun_d(self)
    self.output = true; self.dir[1] = 0;self.dir[2] = 1;self.dir_idx = 3 
  end 
  local function fun_init(self)
    self.output = false
  end 
  local funs =
    {fun_l=fun_l,
     fun_u=fun_u,
     fun_d=fun_d,
     fun_r=fun_r}
  local at_good = Game.at
  local function at(self, x, y)
    if self.time >= complexity + 3 then
      return "q"
    else
      return at_good(self, x, y)
    end
  end
  local time = math.floor((complexity - 2) / 2) + 1
  local function attach(task)
    task.x = task.x + time * task.dir[1]
    task.y = task.y + time * task.dir[2]
    if task[task.x] == nil then
      task[task.x] = {}
    end
    local char = {"l", "u", "d"}
    local c = char[math.random(3)]
    task[task.x][task.y] = c
    local s = {dir = {}}
    funs["fun_" .. c](s)
    task.dir = table.copy(s.dir)
    return task
  end
  local task = {x=1, y=1, dir={1, 0}}
  local task = attach(task)
  task["x"] = nil
  task["y"] = nil
  task["dir"] = nil

  local config = safe_merge(funs, {complexity=complexity,
                            dim=2,
                            init=fun_init,
                            at=at})
  local ret = Game(task, config)
  return ret
end


