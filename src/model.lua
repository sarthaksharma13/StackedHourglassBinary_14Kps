---------------------------------------------------
-- Script to initialize the network architecture --
---------------------------------------------------

-- Load up network model or initialize from scratch
-- opt.netType is either 'hg' or 'hg-binary'paths.dofile('models/' .. opt.netType .. '.lua')
paths.dofile('models/' .. opt.netType .. '.lua')

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)
    model:clearState()
    if opt.nGPU > 1 and torch.typename(model) ~= 'nn.DataParallelTable' then
        require 'nn'
        require 'nngraph'
        require 'nnx'
        require 'cunn'
        require 'cudnn'
        -- -- Optimize GPU memory usage
        -- if opt.optnet then
        --     print('==> Optimizing GPU memory usage')
        --     model = model:cuda()
        --     local optnet = require 'optnet'
        --     local sampleInput = torch.zeros(4, 3, 64, 64):cuda()
        --     optnet.optimizeMemory(model, sampleInput, {inplace = true, mode = 'training'})
        -- end
        
        -- Table of GPU ids to be used for training
        -- Specific GPU ids can be specified (eg. GPUs 1 and 3 as {1,3})
        -- local gpus = {1, 3}
        --local gpus = torch.range(1, opt.nGPU):totable()
        --print("##########################################################################################")
        local gpus = {1,2,3}
        -- Use the fastest conv, according to the CUDNN benchmarks
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
                    :add(model, gpus)
                    :threads(function()
                        require 'nngraph'
                        local cudnn = require 'cudnn'
                        cudnn.fastest, cudnn.benchmark = fastest, benchmark
                    end)
        dpt.gradInput = nil
        model = dpt:cuda()
    end

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    -- By default, the criterion is 'MSE'
    criterion = nn[opt.crit .. 'Criterion']()
end

-- GPU options for the model
if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end
