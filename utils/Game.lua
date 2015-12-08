--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


-- Our problems are executed on a grid (board). Every problem
-- is the instance of this class.

local Game = torch.class('Game')

function init()
  if Game.max_symbols ~= nil then
    return
  end
  Game.max_symbols = 10
  local chars = {"q", "e", "s", "p", "a", "c", "y", "k", "v"}
  Game.dirs = {}
  Game.dirs[1] = {{-1, 0}, {1, 0}} -- Directions to move for 1-dimensional tape.
  Game.dirs[2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}} -- Directions to move for 2-dimensional grid.
  Game.dir_names = {"l", "r", "d", "u"} -- Directions: left, right, down, up.
  Game.encode_map = {}
  Game.decode_map = {}
  for i = 1, Game.max_symbols - 1 do
    Game.decode_map[i] = i
  end
  Game.decode_map[0] = Game.max_symbols
  chars = merge(chars, Game.dir_names)
  for _, c in pairs(chars) do
    Game.decode_map[c] = len(Game.decode_map) + 1
  end
  for k, v in pairs(Game.decode_map) do
    Game.encode_map[v] = k
  end
end

function Game:__init(tab, config)
  init()
  config = config or {}
  for k, v in pairs(config) do
    self[k] = v
  end
  self.A = tab
  self.targets = {}
  self.complexity = 0
  self:reset()
  self:evaluate()
  self:reset()
  self.dim = self.dim or 2
end

function Game:reset()
  self.correct = 0
  self.x = 1
  self.failed = false
  self.y = 1
  self.predicted = {}
  self.strings = {}
  self.target_idx = 1
  self.dir = {1, 0}
  self.output = true
  self.grid = "A"
  self.time = 1
  self.last_prediction_time = 1
  self.normal = #self.targets
  self.dir_idx = 2
  if self.init ~= nil then
    self.init(self)
  end
end

--empty.
function Game:fun_e()
end

function Game:fun_c()
end

function Game:fun_r()
end

function Game:fun_s()
  self.output = false
end

function Game:fun_p()
  self.output = true
end

-- Change direction to the left.
function Game:fun_l()
  self.dir[1] = -1
  self.dir[2] = 0
  self.dir_idx = 1
end

-- Change direction to down.
function Game:fun_d()
  self.dir[1] = 0
  self.dir[2] = 1
  self.dir_idx = 3
end

-- Change direction to up.
function Game:fun_u()
  self.dir[1] = 0
  self.dir[2] = -1
  self.dir_idx = 4
end

-- Change direction to the right.
function Game:fun_r()
  self.dir[1] = 1
  self.dir[2] = 0
  self.dir_idx = 2
end

-- Used to establish targets.
function Game:produce_target()
  if self.output then
    local v = self:at(self.x, self.y)
    if type(v) == "number" then
      table.insert(self.targets, v)
    end
  end
  self.complexity = self.complexity + 1
end

-- Used during model execution. Loss calls
-- this function with it's own predictions. 
-- This function tells if predictions are good or not.
function Game:predict(predicted)
  table.insert(self.predicted, predicted)
  local target = self.targets[self.target_idx] 
  if target == 0 then
    target = params.base 
  end
  local passed = true
  if predicted ~= target then
    passed = false
    self.failed = true
  else
    self.correct = self.correct + 1
  end
  self.last_prediction_time = self.time
  self.target_idx = self.target_idx + 1
  return passed
end

function Game:evaluate()
  while true do
    local c = self:at(self.x, self.y)
    local f = self["fun_" .. c]
    if f ~= nil then
      f(self)
    end
    local ya = 1
    if self.output and type(c) == "number" then
      ya = 2
    end
    self:produce_target()
    self:move()
    if self:at(self.x, self.y) == "q" then
      return
    end
  end
end

function Game:current_input()
  if self:eos() then
    return params.vocab_size
  end
  local v = self:at(self.x, self.y)
  local ret = Game.decode_map[v]
  assert(ret ~= nil)
  return ret
end

function Game:current_target()
  if self:eos() then
    return params.vocab_size
  end
  local ret = Game.decode_map[self.targets[self.target_idx]]
  assert(ret ~= nil)
  return ret
end

-- The board is infinite, and the value in newly visited locations
-- is generated on the flu.
function Game:at(x, y)
  assert(type(x) == "number")
  assert(type(y) == "number")
  if self.A[x] == nil then
    self.A[x] = {}
  end
  if self.A[x][y] == nil then
    self.A[x][y] = math.random(self.max_symbols) - 1
  end
  local t = type(self.A[x][y])
  assert(t == "number" or t == "string")
  return self.A[x][y]
end

-- Used to visualize current state of task. 
function Game:__tostring__()
  local s = "State: x=" .. g_d(self.x) .. ", y=" .. g_d(self.y)
  s = s .. "\ndir=[" .. g_d(self.dir[1]) .. ", " .. g_d(self.dir[2]) .. "]"
  s = s .. "\noutput=" .. tostring(self.output)
  s = s .. "\nScore=" .. tostring(self.correct) .. "/" .. tostring(#self.targets)
  s = s .. "\nLen=" .. tostring(self.complexity)
  s = s .. "\ngrid=" .. tostring(self.grid)
  s = s .. "\n\nA:\n    "
  local from_y = math.min(-5, self.y - 2)
  local to_y = math.max(5, self.y + 2)
  if self.dim == 1 then
    from_y = 1
    to_y = 1
  end
  for y = from_y, to_y do
    for x = math.min(-5, self.x - 2), math.max(5, self.x + 2) do
      local v = self:at(x, y)
      if v == params.base then
        v = 0
      end
      if x == self.x and y == self.y then
        s = s .. green .. v .. reset
      else
        if type(v) == "number" then
          s = s .. v
        else
          s = s .. blue .. v .. reset
        end
      end
    end
    s = s .. "\n    "
  end
  local y = "Y:" 
  local p = "P:"
  for i = 1, #self.targets do
    local t = self.targets[i]
    if t == params.base then
      t = 0
    end
    if i == self.target_idx - 1 then
      y = y .. green .. t .. reset
    else
      y = y .. t
    end
  end
  for i = 1, math.min(self.target_idx - 1, #self.predicted) do
    local t = Game.encode_map[self.predicted[i] ]
    if t == params.base then
      t = 0
    end
    if i == self.target_idx - 1 then
      if t == self.targets[i] then
        p = p .. green .. t .. reset
      else
        p = p .. red .. t .. reset
      end
    else
      p = p .. t
    end
  end
  s = s .. "\n" .. y
  s = s .. "\n" .. p
  local xa = ""
  local ya = ""
  local ta = ""
  local wa = ""
  s = s .. "\nxa" .. "=" .. xa 
  s = s .. "\nya" .. "=" .. ya
  s = s .. "\nta" .. "=" .. ta
  s = s .. "\nwa" .. "=" .. wa
  s = s .. "\nself.failed = " .. tostring(self.failed)
  s = s .. "\nself.target_idx = " .. tostring(self.target_idx)
  for i = 1, (5 - math.max(5, self.y + 2)) do
    s = s .. "\n"
  end
  return s
end

-- Determines if it's the end of the episode.
function Game:eos(execute)
  if self.target_idx > #self.targets or 
     (self:at(self.x, self.y) == "q" and execute == true) or 
     self.time >= self.last_prediction_time + self.complexity + 2 or 
     self.failed then
    return true
  else
    return false
  end
end

-- Moves according to the current move direction.
function Game:move(dir)
  local c = self:at(self.x, self.y)
  dir = dir or self.dir
  self.x = self.x + dir[1]
  self.y = self.y + dir[2]
  self.time = self.time + 1
end


