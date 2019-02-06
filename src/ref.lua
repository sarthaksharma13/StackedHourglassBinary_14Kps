-- Parse command line input and do global variable initialization

-- Torch package (for scientific computing with Lua)
require 'torch'
-- Lua extensions package
require 'xlua'
-- Numeric optimization package for Torch
require 'optim'
-- Torch neural network package
require 'nn'
-- Extensions to the Torch 'nn' package
require 'nnx'
-- Neural network graph package for Torch
require 'nngraph'
-- HDF5 package
require 'hdf5'
-- String operations
require 'string'
-- Image operations
require 'image'
-- Routines to call C code from Lua
ffi = require 'ffi'

-- Set default tensor type
torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
-- projectDir = paths.concat(os.getenv('HOME'),'code', 'hourglass-binary')
projectDir = paths.concat('/tmp', 'hourglass-binary')
--projectDir="/tmp/code/hourglass-experiments"

-- Process command line arguments
paths.dofile('opts.lua')
-- Load helper functions for image
paths.dofile('util/img.lua')
-- Load helper function for evaluation (accuracy metrics, etc)
paths.dofile('util/eval.lua')
-- If the Logger hasn't executed yet, execute it
if not Logger then paths.dofile('util/Logger.lua') end

-- Random number seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end                           

-- Initialize dataset
if not dataset then
    local Dataset = paths.dofile(projectDir .. '/src/util/dataset/' .. opt.dataset .. '.lua')
    dataset = Dataset()
end

-- Global reference (may be updated in the task file below)
if not ref then
    -- Initalize a variable to store the global reference
    ref = {}
    -- Number of output channels (number of keypoints)
    ref.nOutChannels = dataset.nJoints
    -- print(ref.nOutChannels)
    -- Input dimensions ([channels] x [height] x [width]) ??? (or width x height)
    ref.inputDim = {3, opt.inputRes, opt.inputRes}
    -- Output dimensions ([channels] x [height] x [width]) ??? (or width x height)
    ref.outputDim = {ref.nOutChannels, opt.outputRes, opt.outputRes}
end

-- Load up task specific variables / functions (default task is 'pose')
paths.dofile('util/' .. opt.task .. '.lua')

-- Optimization function and hyperparameters
optfn = optim[opt.optMethod]
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        alpha = opt.alpha,
        epsilon = opt.epsilon
    }
end

-- Print out input / output tensor sizes
if not ref.alreadyChecked then
    local function printDims(prefix,d)
        -- Helper for printing out tensor dimensions
        if type(d[1]) == "table" then
            print(prefix .. "table")
            for i = 1,#d do
                printDims("\t Entry " .. i .. " is a ", d[i])
            end
        else
            local s = ""
            if #d == 0 then s = "single value"
            elseif #d == 1 then s = string.format("vector of length: %d", d[1])
            else
                s = string.format("tensor with dimensions: %d", d[1])
                for i = 2,table.getn(d) do s = s .. string.format(" x %d", d[i]) end
            end
            print(prefix .. s)
        end
    end

    printDims("Input is a ", ref.inputDim)
    printDims("Output is a ", ref.outputDim)
    ref.alreadyChecked = true
end
