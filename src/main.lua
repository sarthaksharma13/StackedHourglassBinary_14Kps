
-- Load the 'paths' package, which "provides portable functions to manipulate filenames"
require 'paths'

-- 'dofile' opens the named file and executes its contents as a Lua chunk

-- Parse command line input and do global variable initialization
paths.dofile('ref.lua')
-- Read in network model
paths.dofile('model.lua')
-- Load up training/testing functions
paths.dofile('train.lua')

-- Set the number of threads used by Torch
torch.setnumthreads(1)

-- Set up dataloader
local Dataloader = paths.dofile('util/dataloader.lua')
loader = Dataloader.create(opt, dataset, ref)

-- Initialize logs
ref.log = {}
ref.log.train = Logger(paths.concat(opt.save, 'train.log'), opt.continue)
ref.log.valid = Logger(paths.concat(opt.save, 'valid.log'), opt.continue)

-- Main training loop
for i = 1,opt.nEpochs do
	-- Display message for current epoch
    print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
    

    -- Perform training
    if opt.trainIters > 0 then train() end
    -- Perform validation
    if opt.validIters > 0 then valid() end
    -- Increment the number of elapsed epochs
    epoch = epoch + 1
    -- Perform garbage collection to free unused memory
    collectgarbage()
end

-- -- Update reference for last epoch
-- opt.lastEpoch = epoch - 1

-- -- Save model
-- model:clearState()
-- torch.save(paths.concat(opt.save,'options.t7'), opt)
-- torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
-- torch.save(paths.concat(opt.save,'final_model.t7'), model)


-- -- Generate final predictions on validation set
-- if opt.finalPredictions then
-- 	ref.log = {}
-- 	loader.test = Dataloader(opt, dataset, ref, 'test')
-- 	predict()
-- end
