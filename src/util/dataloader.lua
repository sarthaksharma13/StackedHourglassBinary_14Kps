--
--  Original version: Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--  (Modified a bit by Alejandro Newell)
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--


-- Load the 'threads' module
local Threads = require 'threads'
-- 'sharedserialize' transparently shares the storages, tensors and tds C structures
-- This approach is great if one wants to pass large data structures between threads
Threads.serialization('threads.sharedserialize')

-- Create a DataLoader class
local M = {}
local DataLoader = torch.class('DataLoader', M)


-- Method used to create a DataLoader object
function DataLoader.create(opt, dataset, ref)
   
   -- Table to hold 'train' and 'valid' loaders
   local loaders = {}

   -- For each split (train/val)
   for i, split in ipairs{'train', 'valid'} do
    -- If the split is indeed required (specified in opt.trainIters, opt.validIters)
    if opt[split .. 'Iters'] > 0 then
        -- Call the constructor using the commandline arguments on the specified split 
        -- of the specified dataset. Note that 'ref' contains additional info such as
        -- batch sizes, data dimensions, etc.
        loaders[split] = M.DataLoader(opt, dataset, ref, split)
    end
   end

   -- Return the table of loaders
   return loaders

end


-- Constructor to create the DataLoader for a specific dataset and a specific split
-- Inputs:
-- opt: commandline arguments
-- dataset: name of the dataset being used
-- ref: global reference (contains info such as batch sizes, data dimensions, etc.)
-- split: train/val
function DataLoader:__init(opt, dataset, ref, split)
    
    -- Pre-initialization function (for the thread)
    local function preinit()
        -- Run the dataset specific lua script
        paths.dofile('dataset/' .. opt.dataset .. '.lua')
    end

    -- Initialization function (for the thread)
    local function init()
        -- Initialize the global 'opt', 'dataset', 'ref', and 'split' parameters
        -- (Guessing '_G' is the global env for the current thread ???)
        _G.opt, _G.dataset, _G.ref, _G.split = opt, dataset, ref, split
        -- Run the 'ref.lua' script
        paths.dofile('../ref.lua')
    end

    -- Main function (for the thread)
    local function main(idx)
        -- Specify the number of threads
        torch.setnumthreads(1)
        -- Return the size (number of samples) in the current split
        return dataset:size(split)
    end

    -- Create a pool of 'opt.nThreads' threads, each of which exectues the 
    -- preinit, init, and main functions
    local threads, sizes = Threads(opt.nThreads, preinit, init, main)
    -- Create references within class, to store various variables
    -- The created pool of threads
    self.threads = threads
    -- Number of iterations (per-epoch)
    self.iters = opt[split .. 'Iters']
    -- Batchsize
    self.batchsize = opt[split .. 'Batch']
    -- Number of samples (???)
    self.nsamples = sizes[1][1]
    -- Split (train/val)
    self.split = split
end


-- Get the number of iterations (per-epoch) for the current split
function DataLoader:size()
    return self.iters
end


-- Run method, used when the DataLoader actually fetches data for the network
function DataLoader:run()

    -- Pool of threads
    local threads = self.threads
    -- Total number of images required to be fetched
    local size = self.iters * self.batchsize

    -- Indices of all samples
    local idxs = torch.range(1, self.nsamples)
    -- ??? (Replicating idxs for epochs)
    for i = 2,math.ceil(size/self.nsamples) do
        idxs = idxs:cat(torch.range(1,self.nsamples))
    end
    -- Shuffle indices
    idxs = idxs:index(1,torch.randperm(idxs:size(1)):long())
    -- Map indices to training/validation/test split
    idxs = opt.idxRef[self.split]:index(1,idxs:long())

    local n, idx, sample = 0, 1, nil
    -- Enqueue an image
    local function enqueue()
        -- If there are images yet to be passed this epoch, and a thread is ready
        while idx <= size and threads:acceptsjob() do
            -- Compute the indices of the images in the next batch
            local indices = idxs:narrow(1, idx, math.min(self.batchsize, size - idx + 1))
            -- Add the job to the thread
            threads:addjob(
                function(indices)
                    local inp,out = _G.loadData(_G.split, indices)
                    collectgarbage()
                    return {inp,out,indices}
                end,
                function(_sample_) sample = _sample_ end, indices
            )
            -- Increment the index of the next image that is to be passed
            idx = idx + self.batchsize
        end
    end

    -- Main loop
    local function loop()
        
        -- Enqueue a job
        enqueue()
        -- If the thread does not have a job, return
        if not threads:hasjob() then return nil end
        -- If the thread has a job, do it (call the loadData function)
        threads:dojob()
        -- If there is a thread error, synchronize
        if threads:haserror() then threads:synchronize() end
        -- Enqueue a job
        enqueue()
        -- Increment the number of calls
        n = n + 1

        return n, sample
    end

    return loop
end

-- Return the DataLoader object
return M.DataLoader
