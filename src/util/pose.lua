-- Task-specific variables and functions for the 'pose' task

-- Update dimension references to account for intermediate supervision
-- Usually, we apply the loss at 5 points. ???
ref.predDim = {dataset.nJoints, 5}

-- Dimensionality of the output layer
ref.outputDim = {}
-- Add a ParallelCriterion (weighted sum of other Criterion)
criterion = nn.ParallelCriterion()
-- For each hourglass that is being stacked
for i = 1,opt.nStack do
    -- Add an entry in ref.outputDim
    ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
    -- Add the Criterion specified in 'opt' (default is MSE)
    criterion:add(nn[opt.crit .. 'Criterion']())
end


-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x)) end


-- Code to generate training samples from raw images
-- Inputs
-- set: train/val/test
-- idx: index of the sample in the dataset
-- Outputs
-- img: image
-- out: ground-truth heatmap
function generateSample(set, idx)

    -- Load the image corresponding to the sample whose index is specified
    local img = dataset:loadImage(idx)
    -- Get the keypoints, center, and scale of the sample
    local pts, c, s = dataset:getPartInfo(idx)

    -- Variable to store the amount of rotation to be applied
    local r = 0

    if set == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
    end

    -- Crop the input image
    local inp = crop(img, c, s, r, opt.inputRes)
    -- Initialize the output image (label) to a blank one
    local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
    -- Draw a Gaussian over the ground-truth kp coordinates
    for i = 1,dataset.nJoints do
        if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(pts[i], c, s, r, opt.outputRes), opt.hmGauss)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    return inp, out
end


-- Load in a mini-batch of data
-- set: train/val/test
-- idxs: indices from which samples are to be drawn
function loadData(set, idxs)

    -- If indices are stored in a table, convert them to a torch Tensor
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    -- Number of samples to be drawn
    local nsamples = idxs:size(1)

    -- Variables to store the images and labels (for the entire batch)
    local input, label

    -- For each sample
    for i = 1,nsamples do
        -- Variables to store an image and its label
        local tmpInput, tmpLabel
        -- Generate a sample (perform color space augmentation, random flips, random crops)
        tmpInput, tmpLabel = generateSample(set, idxs[i])
        -- Convert the tensor to a table
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        -- Concatenate them to 'input' and 'label' respectively
        if not input then
            input = tmpInput
            label = tmpLabel
        else
            input = input:cat(tmpInput,1)
            label = label:cat(tmpLabel,1)
        end
    end

    -- For each hourglass that is stacked, restack the label
    if opt.nStack > 1 then
        -- Set up label for intermediate supervision
        local newLabel = {}
        for i = 1,opt.nStack do newLabel[i] = label end
        return input,newLabel
    else
        return input,label
    end
end


-- ???
function postprocess(set, idx, output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    local p_tf = torch.zeros(p:size())
    for i = 1,p:size(1) do
        _,c,s = dataset:getPartInfo(idx[i])
        p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
    end
    
    return p_tf:cat(p,3):cat(scores,3)
end


-- Compute accuracy
function accuracy(output,label)
    if type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs)
    end
end
