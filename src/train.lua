-- Variables and functions required for training the network


-- Prepare tensors for saving network output

-- Number of validation samples
local validSamples = opt.validIters * opt.validBatch
-- Initialize tensors to hold network output that has to be saved
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end


-- Main processing step
-- Input
-- tag: 'train' or 'valid' or 'predict'
function step(tag)

    -- Variables to store Average loss, Average accuracy
    local avgLoss, avgAcc = 0.0, 0.0
    -- Variables to store network output, error, index
    local output, err, idx
    -- Returns the flattened learnable params and the gradients of the energy wrt these params
    local param, gradparam = model:getParameters()
    -- Function to evaluate an input x and return the network output and the gradients ???
    local function evalFn(x) return criterion.output, gradparam end

    -- In training mode
    if tag == 'train' then
        -- Set the mode of the Module (or sub-modules) to 'train = true'
        -- This mode is useful for modules like BatchNorm that have a different behavior during training
        -- and evaluation
        model:training()
        -- We will work with the train split
        set = 'train'
    else
        -- Set the mode of the Module (or sub-modules) to 'train = false'
        -- This mode is useful for modules like BatchNorm that have a different behavior during training
        -- and evaluation
        model:evaluate()

        -- In test mode
        if tag == 'predict' then
            -- Generate predictions over the test set
            print("==> Generating predictions...")
            -- Number of samples in the test set
            local nSamples = dataset:size('test')
            -- Variable to store the indices and predictions of test data
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            -- We will work with the test split
            set = 'test'
        else
            -- We will work with the val split
            set = 'valid'
        end
    end

    -- Get the number of iterations to run on the current set (train/valid/test)
    local nIters = opt[set .. 'Iters']

    -- For each sample in the current set (batch ???)
    for i,sample in loader[set]:run() do

        -- Display the progress bar
        xlua.progress(i, nIters)
       
        -- Unpack the sample to various tensors
        local input, label, indices = unpack(sample)


        --print(label) A tabel of 8 :  batchsize x nkeyPs x 64x 64

        -- In GPU mode
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end

        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        -- print(output)
        -- print(label)
        local err = criterion:forward(output, label)
        avgLoss = avgLoss + err / nIters

        -- Training: Do backpropagation and optimization
        if tag == 'train' then
            -- ??? Clear any gradients that have been accumulated at modules during a previous backward pass
            model:zeroGradParameters()
            -- Perform a backward pass (accumulate gradients)
            model:backward(input, criterion:backward(output, label))
            -- Update parameters using the optimization method (default: 'rmsprop')
            optfn(evalFn, param, optimState)
        -- Validation: Get flipped output
        else
            output = applyFn(function (x) return x:clone() end, output)
            -- local flippedOut = model:forward(flip(input))
            -- flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            -- output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

            -- Save sample
            local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))
        end

        
        avgAcc = avgAcc + accuracy(output, label) / nIters
    end


    -- Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end


function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
