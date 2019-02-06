-- Functions to initialize parameters specific to the Pascal3D dataset


-- Define a local variable M
local M = {}
-- Define a Torch class, inheriting from M. Class will be accessible through the table M,via M.Dataset
Dataset = torch.class('pose.Dataset', M) 

-- initializing the class
function Dataset:__init()

    -- Number of keypoints in the dataset
    self.nJoints = 14
    -- 1 : 'L_F_WheelCenter'  2 : 'R_F_WheelCenter'  3 : 'L_B_WheelCenter' 4 : 'R_B_WheelCenter' 5 : 'L_HeadLight'
    -- 6 : 'R_HeadLight' 7 : 'L_TailLight' 8 : 'R_TailLight' 9 : 'L_SideviewMirror' 10 : 'R_SideviewMirror'
    -- 11 : 'L_F_RoofTop' 12 : 'R_F_RoofTop' 13 : 'L_B_RoofTop' 14 : 'R_B_RoofTop'

    -- Keypoints for which accuracy is computed
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14}

    -- Keypoints that occur as reflecting pairs.
    self.flipRef = {{1,2},   {3,4},   {5,6},
                    {7,8}, {9,10}, {11,12},{13,14}}

    --[[
    -- Pairs of joints for drawing skeleton (assuming this is (joint 1, joint 2, color)) ???
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}
                        ]]
    --[[

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
    ]]

    local annot ={}
    local tags={'index','imgname','part','istrain'}
    -- index/imgname : index/name of the image
    -- part :  keypointsfor image : check getpartInfo function
    
    -- visible : visible keypoints ,used in eval.lua ???
    -- istrain : whether a train image or test image. 1 if true

    --datapath = paths.concat('/home/data/datasets/PASCAL3D/dataHG64_Original.txt')
    
    datapath = paths.concat('/home/data/datasets/PASCAL3D/dataHG64.txt')
    

    --[[
    for line in io.open(datapath) do
        cindex,cimgname,cpart,cnormalize,cvisible,cistrain=unpack(line:split(" "))
        table.insert(annot[index],tonumber(cindex))
        table.insert(annot[imgname],cimgname)
        
        --  made into a table
        sep=","
        tmp= {cpart:match((cpart:gsub("[^"..sep.."]*"..sep, "([^"..sep.."]*)"..sep)))}
        for i=1,#tmp do
            table.insert(annot[part],tonumber(tmp[i]))
        end
        table.insert(annot[part],tonumber(cpart:sub(#cpart-1,#cpart)))
                
        table.insert(annot[normalize],tonumber(cnormalize))
        table.insert(annot[istrain],tonumber(cistrain))       
    end
    ]]

    local totalimgs=0;
    for line in io.lines(datapath) do 
        totalimgs = totalimgs+1;
    end

    local indextensor=torch.Tensor(totalimgs)
    local imgnametensor = torch.Tensor(totalimgs)
    -- local imgnametable={}; -- ??? saving image names in a table,as string tensors dont exist.
    local parttensor=torch.Tensor(totalimgs,14,2)-- ??1
    local istraintensor=torch.Tensor(totalimgs)
    local sep;
    local str;

    local i=1;
    for line in io.lines(datapath) do
        cindex,cimgname,cpart,cistrain=unpack(line:split(" "));
        
        indextensor[i]=tonumber(cindex);

        sep='/';
        str= {cimgname:match((cimgname:gsub("[^"..sep.."]*"..sep, "([^"..sep.."]*)"..sep)))}
        -- table.insert(imgnametable,str[#str])
        tempFilename = str[#str]
        -- tempFilename:strsub(1, tempFilename:len()-4))
        imgnametensor[i] = tonumber(tempFilename:sub(1, tempFilename:len()-4))

        -- Make the co-ordinate string into a 14x2 tensor and insert for the ith image.
        local temp = torch.DoubleTensor(14,2)
        sep=","
        -- A table of size 28x1 ,put an extra comma after the last co-ordinate to make sure 
        -- it also comes in the table.
        str= {cpart:match((cpart:gsub("[^"..sep.."]*"..sep, "([^"..sep.."]*)"..sep)))}
        local rw=1
        for j=1,#str-1,2 do
            -- temp[rw][1]=tonumber(str[j])
            -- temp[rw][2]=tonumber(str[j+1])
            -- IMP: Flipping X and Y coordinates, since the traindata file contains Y,X pairs
            temp[rw][1] = tonumber(str[j])
            temp[rw][2] = tonumber(str[j+1])
            rw=rw+1;
        end
        parttensor[i]=temp
        -- check for visible ??
        
    
        istraintensor[i]=tonumber(cistrain);

        i=i+1;

    end
    annot['index']=indextensor;
    -- annot['imgname']=imgnametable;
    annot['imgname'] = imgnametensor
    annot['part']=parttensor
    annot['istrain']=istraintensor;

    --[[  annot.index:add(1)
    annot.person:add(1)
    annot.part:add(1)
    ]]





    -- Index reference
    if not opt.idxRef then
        -- Get all the indices
        -- This methods is not for table
        local allIdxs = torch.range(1,annot.index:size(1))
        -- Table to store index reference
        opt.idxRef = {}
        -- Test indices : set 0
        -- This methods is not for table
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        -- Train indices : set 1

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


-- Get the size of the dataset for given mode (train,test,valid)
function Dataset:size(set)
    return self.nsamples[set]
end


-- Get path to the sample with a particular index
function Dataset:getPath(idx)
   
    --return paths.concat('/home/data/datasets/PASCAL3D/dataHG64_Original', tostring(self.annot.imgname[idx]) .. '.jpg')
   
    return paths.concat('/home/data/datasets/PASCAL3D/dataHG64', tostring(self.annot.imgname[idx]) .. '.jpg')
   
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
    return pts
end

--[[
-- Get the normalization constants for the sample with a specified index
function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end
]]

-- Return the dataset table
return M.Dataset

