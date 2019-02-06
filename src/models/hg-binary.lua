-- Defines the architecture for the hourglass model
-- We use a unary loss (per keypoint likelihood) and binary losses (relative kp loc of 
-- a kp wrt an anchor)

require 'cutorch'
require 'cunn'

-- Load the Residual block module
paths.dofile('layers/Residual.lua')


-- Describes the hourglass module
-- Inputs
-- n: number of 'stages' in the hourglass
-- f: number of featurs to be maintained across the hourglass layers
-- inp: network architecture upto the layer just before the hourglass
local function hourglass(n, f, inp)
    
    -- Upper branch (the branch without size reduction)
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch (the branch that involves pooling)
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    

    local low2
    -- Recursively call the function, to add layers in the hourglass
    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    -- Upsampling modules
    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end


-- Apply a 1 x 1 convolution, followed by a BatchNorm and a ReLU
-- Inputs
-- numIn: number of input channels
-- numOut: number of output channels
-- inp: input computational graph
local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end


-- Function to create the hourglass model
function createModel()

    local inp = nn.Identity()()

    -- -- Initial processing of the image
    -- local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    -- local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    -- local r1 = Residual(64,128)(cnv1)
    -- local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    -- local r4 = Residual(128,128)(pool)
    -- local r5 = Residual(128,opt.nFeats)(r4)

    
    ---
    -- Testing out a network that takes in 3 x 64 x 64 images as input
    ---

    local cnv1_ = nnlib.SpatialConvolution(3,64,3,3,1,1,1,1)(inp)
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local r2 = Residual(128,128)(r1)
    local r5 = Residual(128,opt.nFeats)(r2)

    ---
    ---


    local out = {}
    local inter = r5

    -- For each hourglass to be stacked together
    for i = 1,opt.nStack do

        -- Add an hourglass module
        local hg = hourglass(4, opt.nFeats, inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats, opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats, opt.nFeats, ll)

        -- Predicted heatmaps
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll)
        -- Multiply each heatmap by -1
        local tmpOut2 = nn.MulConstant(-1, false)(tmpOut)
        -- Select the anchor heatmap (we use the first keypoint as the anchor)
        -- ### Verify if we have to indeed select along the second dimension
        local anchorMap = nn.Select(2, 1)(tmpOut)
        -- ### Verify if we have to indeed replicate along the second dimension
        anchorMap = nn.Replicate(ref.nOutChannels,2)(anchorMap)
        -- Get the relative keypoint prediction error heatmap (subtract the anchor kp
        -- likelihood from each kp likelihood)
        relativeMap = nn.CAddTable()({anchorMap, tmpOut2})
        -- Throw away the anchor dimension (since it will always be a Tensor of zeros)
        -- relativeMap = nn.NarrowTable(2,ref.nOutChannels-1)(relativeMap)
        
        table.insert(out, tmpOut)
        table.insert(out, relativeMap)

        -- Add predictions back to the network
        if i < opt.nStack then
            -- Unary predictions
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            -- Binary (relative) predictions
            local relativeMap_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(relativeMap)
            -- Add all of these back to the network
            inter = nn.CAddTable()({inter, ll_, tmpOut_, relativeMap_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)
    -- print('---------------');

    -- Data Parallelizing the model across multiple GPUs (buggy code ???)
    if opt.nGPU > 1 then

        -- Optimize GPU memory usage
        if opt.optnet then
            print('==> Optimizing GPU memory usage')
            model = model:cuda()
            local optnet = require 'optnet'
            local sampleInput = torch.zeros(4, 3, 64, 64):cuda()
            optnet.optimizeMemory(model, sampleInput, {inplace = true, mode = 'training'})
        end
        
        -- Table of GPU ids to be used for training
        -- Specific GPU ids can be specified (eg. GPUs 1 and 3 as {1,3})
        -- local gpus = {1, 3}
        --local gpus = torch.range(1, opt.nGPU):totable()
        --print("##########################################################################################")
        local gpus ={1,2};
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

    -- Save the model to a fig file
    -- graph.dot(model.fg, 'hg-binary', '/home/km/code/hourglass-binary/exp/arch')

    return model

end
