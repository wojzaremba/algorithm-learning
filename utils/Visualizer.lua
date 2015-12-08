--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--


local Visualizer = torch.class('Visualizer')

function Visualizer:__init()
  self.last_visualize = nil
end

function Visualizer:get_global_desc()
  beginning_time = beginning_time or torch.tic()
  local total_cases = model.step * params.seq_length * params.batch_size
  local wps = math.floor(total_cases / torch.toc(beginning_time))
  since_beginning = torch.toc(beginning_time) / 60
  local pr = {tostring(acc)}
  table.insert(pr, "norm of weights=" .. g_f5(paramx:norm()) ..
                   ", norm of gradient=" .. g_f5(model.norm_dw) ..
                   ", number of parameters=" .. g_d(paramx:size(1)) ..
                   ', characters per second=' .. wps .. 
                   ', since beginning=' .. g_d(since_beginning) .. ' mins.' ..
                   ', lr=' .. g_f3(params.lr))

  local sizes = ""
  table.insert(pr, 'current complexity=' .. tostring(curriculum.complexity) ..
                   ', time-step=' .. tostring(model.step) ..
                   ', trained characters=' .. tostring(model.trained_chars) ..
                   ', number of unrolling steps=' .. tostring(params.seq_length))
  local counts = "Number of training samples so far: " .. curriculum.type_count
  table.insert(pr, counts)
  table.insert(pr, "")
  local s = ""
  for i = 1, #pr do
    s = s .. pr[i] .. "\n"
  end
  return s
end

function Visualizer:get_sample_desc(sample)
  local desc =  "task_name=" .. tostring(sample.task_name) ..
          ", complexity=" .. tostring(sample.complexity) .. "\n\n"
  return desc
end

function Visualizer:save_movie(sample)
  if sample == nil then
    return
  end
  name = sample.task_name
  os.execute("mkdir -p movie")
  os.execute("rm -rf movie/" .. name .. "*")
  local begin = sample.ins
  local idx = sample.idx
  local from = model(begin)
  local global = self:get_global_desc()
  local desc   = self:get_sample_desc(sample)
  for i, s in pairs(sample.strings) do
    local f = io.open(string.format("movie/%s_%d", name, i), "w")
    f:write(global)
    f:write(desc)
    f:write(s)
    f:write("\n\n")
    local ins = model(from.time + i - 1)
    f:write(self:actions_desc(ins, idx))
    f:write(self:instance_desc(ins, idx))
    f:close()
  end
end

function Visualizer:visualize(force)
  if force or self.last_params == nil or torch.toc(self.last_params) > 20 then
    last_params = torch.tic()
    print(params)
    print("")
  end
  print(self:get_global_desc())
  if model.step % 20 ~= 1 or 
     params.train == 1 then
    return
  end
  local random_sample = nil
  for _, sample in pairs(model.samples) do
    if sample:eos() and sample.train == 1 then
      local ins = model(sample.ins)
      local begin = ins.data.begin[sample.idx] 
      local finish = ins.data.finish[sample.idx] 
      assert(begin >= 1 and finish <= params.seq_length)
      if finish - begin >= 0 then
        local acc = sample.correct / sample.normal
        random_sample = sample
        break
      end
    end
  end
  self:save_movie(random_sample)
end


function Visualizer:actions_desc(ins, idx)
  local s = ""
  local sample = ins.data.samples[idx]
  s = s .. "\nlogits: \n"
  for pred = 1, ins.q:size(2) do
    local a = interfaces:decode_q(pred)
    for k, v in pairs(a) do
      s = s .. k .. tostring(v) .. "," 
    end
    local max_idx = ins.max_idx[q]
    if max_idx == pred then
      s = s .. blue
    end
    s = s .. string.format("logits = %.2f, ", ins.logits[idx][pred])
    if max_idx == pred then
      s = s .. reset
    end
    s = s .. "\n"
  end
  s = s .. string.format("reward = %d", ins.correct[idx])
  return s
end

function Visualizer:instance_desc(ins, idx)
  local from = ins.data.begin[idx]
  local to = ins.data.finish[idx]
  return string.format("\nfrom:%d, to:%d, ins:%d", from, to, ins.time)
end

function short_prob(x)
  local function number2short_prob(x)
    assert(x >= 0 and x <= 1)
    if x == 0 then
      return "0"
    elseif x > 0.05 and x < 0.95 then
      return tostring(math.floor(10 * x + 0.5))
    elseif x >= 0.95 then 
      return "9" 
    else
      return string.char(string.byte('a') + math.min(math.floor(-math.log(x)) - 1, 20))
    end
    assert(false)
  end
  local ret = ""
  assert(math.abs(x:sum() - 1) < 1e-3, tostring(x))
  for i = 1, x:size(1) do
    ret = ret .. number2short_prob(x[i])
  end
  return ret
end



