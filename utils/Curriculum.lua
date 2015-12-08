--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


local Curriculum = torch.class('Curriculum')

function Curriculum:__init()
  self.min_complexity = 6
  self.complexity = self.min_complexity
  self.type_count = 0 -- Number of trained examples.
  self.min_min_acc = 0.95
  params.train = 2
end

function Curriculum:progress()
  -- Checks if achieved good results on test data.
  if params.train == 1 then
    if acc:is_good() then
      if acc:is_normal() then
        Visualizer:visualize(true)
        print("DONE")
        print(acc)
        os.exit(0)
      end
    else
      self.complexity = self.complexity + 4
      self.complexity = math.min(self.complexity, params.test_len - 10)
      params.train = 2
      model.instances = {}
      return
    end
  end
  if acc:is_good(self.min_min_acc) and
     acc:is_normal() then
    -- When the training seems to work reasonably well, then we start penalizing Q(s, \bullet).
    params.q_decay_lr = 0.01
    params.decay_expr = 1
  end
  if not acc:is_normal() or not acc:is_good() then
    return
  end
  params.train = 1
  Visualizer:visualize(true)
  model:clean()
end

-- Chooses complexity for a sample.
function Curriculum:pick_complexity()
  local complexity = self.complexity - math.random(5) + 1
  complexity = math.max(complexity, 2)
  local acc_current = acc:get_current_acc()
  -- Complexity for a sample when we test.
  if params.train == 1 then
    local pow = math.log(params.test_len - 6 - self.complexity) / math.log(2)
    pow = math.floor(pow)
    complexity = self.complexity + math.pow(2, pow - math.random(3) + 1) + 5
  end
  return complexity
end

-- Dynamically decides on the number of unrolling steps.
function Curriculum:expandLength(sample)
  if params.train == 2 then
    params.seq_length = math.max(params.seq_length or 1, sample.complexity + 5)
    params.seq_length = math.min(params.seq_length, params.max_seq_length - 1)
  else
    params.seq_length = math.max(params.seq_length, 20)
  end
end

function Curriculum:generateNewSample()
  local complexity = self:pick_complexity()
  local sample = data_generator["task_" .. params.task](data_generator, complexity)
  sample.task_name = params.task
  sample.train = params.train
  -- 20% of samples from training are used for the validation. 
  -- The real testing occurs on much longer samples.
  if math.random(5) == 1 then
    sample.train = 1
  end
  self:expandLength(sample)
  -- Counts number of used samples so far.
  self.type_count = self.type_count + 1
  return sample
end

