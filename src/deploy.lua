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

--Caliing the network for validation
local validSamples = opt.validIters * opt.validBatch

valid()


--[[
--predsfile ='/home/sarthaksharma/code/hourglass-experiments/exp/pascal3D/test/preds.h5'
predFilename='preds.h5'
preds = hdf5.open(paths.concat(opt.save,predFilename))


for i=1,1 do

    -- Heatmaps
  hms = preds:read('heatmaps'):partial({i,i},{1,14},{1,64},{1,64})
  hms=hms[1]
  first=hms[1]
  image.save('/home/sarthaksharma/heatmap_1KP_'..tostring(index)..'.jpg',first)

    -- Index of image
    index=preds:read('idxs'):partial({i,i})
    index=index[1]

  -- Input Image
  img=preds:read('input'):partial({i,i},{1,3},{1,64},{1,64})
  img=img[1]
  image.save('/home/sarthaksharma/image_1KP_'..tostring(index)..'.jpg',first)
  -- Check from the image in the data folder
    -- check=image.load('/home/data/datasets/PASCAL3D/dataHG64/'..tostring(index)..'.jpg')
    -- image.save('/home/sarthaksharma/check_'..tostring(index)..'.jpg',check)

  -- Keypoints
  old_coords = preds:read('preds'):partial({i,i},{1,14},{1,2}) 

  -- Displayin the image with its coordinates






end
]]
