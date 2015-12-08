--[[
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
]]--

require('nn')
require('torch')
require('nngraph')
require('optim')
include("instance/Model.lua")
include("instance/Instance.lua")
include("layers/Index.lua")
include("layers/Loss.lua")
include("layers/Move.lua")
include("layers/EpisodeBarrier.lua")
include("interfaces/Interface.lua")
include("interfaces/Interfaces.lua")
include("interfaces/Data.lua")
include("utils/base.lua")
include("utils/Acc.lua")
include("utils/nngraph.lua")
include("utils/models.lua")
include("utils/Visualizer.lua")
include("utils/DataGenerator.lua")
include("utils/Curriculum.lua")
include("utils/Game.lua")
include("utils/colors.lua")
include("utils/q_learning.lua")

lapp = require 'pl.lapp'

params = lapp[[
  --batch_size       (default 20)
  --task             (default "addition")   copy | reverse | walk | addition | addition3 | single_mul
  --seed             (default 1)            random initialization seed
  --q_type           (default "q_watkins")  q_classic | q_watkins
  --q_discount       (default -1)           -1 (dynamic discount) | 0.95 (discount of 0.95) | 1 (no discount)
  --q_lr             (default 0.1)          learning rate over q-function
  --unit             (default "gru")        feedforward | lstm | gru
  --test_len         (default 200)          complexity of the test instances
  --max_seq_length   (default 50)           maximum complexity of the training instances
  --layers           (default 1)            number of layers
  --rnn_size         (default 200)          number of hidden units
  --lr               (default 0.1)          learning rate
  --max_grad_norm    (default 5)            clipping of gradient norm
]]

if params.task == "reverse" or
   params.task == "ident" then
  params.dim = 1
else
  params.dim = 2
end

function create_network()
  interfaces:set_sizes()
  g_make_deterministic(params.seed)
  local ins_node_org             = nn.Identity()()
  local prev_s                   = nn.Identity()()
  local s                        = prev_s
  local ins_node                 = ins_node_org
  -- Ensures that state doesn't leak between consecutive samples.
  s                              = nn.EpisodeBarrier()({ins_node, s})
  local embs                     = {}
  -- Input from the all interfaces.
  for name, interface in pairs(interfaces.interfaces) do
    embs                         = merge(embs, interface:last_action(ins_node))
    embs                         = merge(embs, interface:view(ins_node))
  end
  local join                     = join_table(embs)
  local linear                   = nn.Linear(params.input_size, params.rnn_size)
  -- It's an LSTM, GRU, or FF.
  local h, next_s                = _G[params.unit](linear(join), s)
  h                              = nn.Linear(params.rnn_size, params.rnn_size)(h)
  -- Computes Q-function.
  ins_node                       = interfaces:q_learning(ins_node, h)
  ins_node                       = interfaces:apply(ins_node)
  local pre_prob                 = nn.Linear(params.rnn_size, params.vocab_size)(nn.Tanh()(h))
  local prob                     = nn.LogSoftMax()(pre_prob)
  local target                   = interfaces.data:target(ins_node)
  -- Computes loss.
  ins_node                       = nn.Loss()({ins_node, prob, target})
  ins_node                       = nn.Move()(ins_node)
  return nn.gModule({ins_node_org, prev_s}, {ins_node, next_s})
end

function update_weights()
  -- Gradient clipping.
  local norm_dw = paramdx:norm()
  if norm_dw ~= norm_dw or norm_dw >= 10000 then
    print("\nNORM TOO HIGH", norm_dw)
    os.exit(-1)
  end
  local shrink_factor = 1
  if norm_dw > params.max_grad_norm then
    shrink_factor = params.max_grad_norm / norm_dw
  end
  model.norm_dw = norm_dw
  paramdx:mul(shrink_factor)
  return 0, paramdx
end

function setup()
  torch.setdefaulttensortype('torch.FloatTensor')
  g_make_deterministic(params.seed)
  params.vocab_size = 24
  initial_params = {}
  for k, v in pairs(params) do
    initial_params[k] = v
  end
  -- Generates tasks.
  data_generator = DataGenerator()
  -- Interfaces (i.e. tape, grid).
  interfaces = Interfaces()
  -- Stores model parameters.
  model = Model()
  -- Provides curriculum over the samples.
  curriculum = Curriculum()
  -- Keeps track of what have been solved.
  acc = Acc()
  -- Visualizes execution.
  visualizer = Visualizer()
  model:reboot()
end

setup()
print("Network parameters:")
print(params)
print("Starting training.")
while true do
  model:fp()
  model:bp()
  model.step = model.step + 1
  curriculum:progress()
  optim.sgd(update_weights, paramx, {learningRate=params.lr})
  if model.step % 30 == 0 then
    collectgarbage()
  end
  if curriculum.complexity > params.max_seq_length then
    os.exit(0)
  end
end
