-- Functions to initialize parameters specific to the MPII dataset


-- Define a local variable M
local M = {}
-- Define a Torch class, inheriting from M
Dataset = torch.class('pose.Dataset', M)

-- (kind-of) constructor
function Dataset:__init()

    -- Number of joints in the dataset
    self.nJoints = 16
    -- Indices of keypoints whose accuracies we're interested in
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    -- ???
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton (assuming this is (joint 1, joint 2, color)) ???
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    -- Table to store annotations
    local annot = {}
    -- Various tags available in the hdf5 files ???
    local tags = {'index','person','imgname','part','center','scale',
                  'normalize','torsoangle','visible','multi','istrain'}
    -- Read in all annotations
    local a = hdf5.open(paths.concat(projectDir,'data/mpii/annot.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()

    -- Torch indexing starts from 1, as opposed to 0
    annot.index:add(1)
    annot.person:add(1)
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        -- Get all the indices
        local allIdxs = torch.range(1,annot.index:size(1))
        -- Table to store index reference
        opt.idxRef = {}
        -- Test indices
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        -- Train indices
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]

        -- Set up training/validation split
        local perm = torch.randperm(opt.idxRef.train:size(1)):long()
        -- Final val indices
        opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
        -- Final train indices
        opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))

        -- Save the options to a .t7 file
        torch.save(opt.save .. '/options.t7', opt)
    end

    -- Variable to store the annotations
    self.annot = annot
    -- Number of train, val, and test indices
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end


-- Get the size of the dataset
function Dataset:size(set)
    return self.nsamples[set]
end


-- Get path to the sample with a particular index
function Dataset:getPath(idx)
    return paths.concat('/home/data/datasets/mpii', 'images', ffi.string(self.annot.imgname[idx]:char():data()))
    -- return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end


-- Load an image with a particular index
function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end


-- Get part information for the sample with a specified index
function Dataset:getPartInfo(idx)
    -- Keypoint locations
    local pts = self.annot.part[idx]:clone()
    -- Center location
    local c = self.annot.center[idx]:clone()
    -- Scale (with respect to 200 px height)
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    c[2] = c[2] + 15 * s
    s = s * 1.25
    return pts, c, s
end


-- Get the normalization constants for the sample with a specified index
function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end


-- Return the dataset table
return M.Dataset

