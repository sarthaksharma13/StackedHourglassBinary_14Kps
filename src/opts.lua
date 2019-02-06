-- Parse command-line arguments

if not opt then

-- projectDir = projectDir or paths.concat(os.getenv('HOME'),'code', 'hourglass-binary')
projectDir = paths.concat('/tmp', 'hourglass-binary')

local function parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID',       'test', 'Experiment ID')
    cmd:option('-dataset',    'pascal3D', 'Dataset choice: mpii | flic | pascal3D')
    cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
    -- cmd:option('-dataDir',  '/home/data/datasets/mpii', 'Data directory')
    cmd:option('-imageDir', '/home/data/datasets/PASCAL3D/dataHG64', 'Image directory')
    cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('-manualSeed',         -1, 'Manually set RNG seed')
    cmd:option('-GPU',                 3, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nGPU',                2, 'Number of GPUs to be used')
    cmd:option('-optnet',                true, 'Use optnet to reduce GPU memory usage')
    cmd:option('-finalPredictions',false, 'Generate a final set of predictions at the end of training (default no)')
    cmd:option('-nThreads',            4, 'Number of data loading threads')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',          'hg-binary', 'Options: hg | hg-binary')
    cmd:option('-loadModel', 'none', 'Provide full path to a previously trained model')
    -- cmd:option('-loadModel',      '/home/km/code/hourglass-binary/exp/pascal3D/hg-car-kps-pascal.t7', 'Provide full path to a previously trained model')
    cmd:option('-continue',        false, 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off')
    -- The task to be performed (a Lua script with this name must be present in the 'utils' dir)
    -- cmd:option('-task',           'pose', 'Network task: pose | pose-int')
    cmd:option('-task',      'pascalKps', 'Network task: pose | pose-int | pascalKps')
    cmd:option('-nFeats',            256, 'Number of features in the hourglass')
    cmd:option('-nStack',              8, 'Number of hourglasses to stack')
    cmd:option('-nModules',            1, 'Number of residual modules at each location in the hourglass')
    cmd:text()
    cmd:text(' ---------- Snapshot options -----------------------------------')
    cmd:text()
    cmd:option('-snapshot',            1, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-saveInput',       true, 'Save input to the network (useful for debugging)')
    cmd:option('-saveHeatmaps',    true, 'Save output heatmaps')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-LR',             2.5e-5, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-alpha',            0.99, 'Alpha')
    cmd:option('-epsilon',          1e-8, 'Epsilon')
    cmd:option('-crit',            'MSE', 'Criterion type')
    cmd:option('-optMethod',   'sgd', 'Optimization method: rmsprop | sgd | nag | adadelta')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-nEpochs',          1, 'Total number of epochs to run')
    cmd:option('-trainIters',       2, 'Number of train iterations per epoch')
    cmd:option('-trainBatch',       14, 'Mini-batch size')
    cmd:option('-validIters',       17, 'Number of validation iterations per epoch')
    cmd:option('-validBatch',       14, 'Mini-batch size for validation')
    cmd:option('-nValidImgs',      238, 'Number of images to use for validation')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',          64, 'Input image resolution')
    cmd:option('-outputRes',          64, 'Output heatmap resolution')
    cmd:option('-scale',             .25, 'Degree of scale augmentation')
    cmd:option('-rotate',             30, 'Degree of rotation augmentation')
    cmd:option('-hmGauss',             1, 'Heatmap gaussian size')

    local opt = cmd:parse(arg or {})
    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
    -- opt.save = paths.concat(opt.expDir, opt.expID)
    opt.save = '/home/sarthaksharma/code/hourglass-binary/exp/opt.expID'
    -- print(opt.nGPU)
    return opt
end

-------------------------------------------------------------------------------
-- Process command line options
-------------------------------------------------------------------------------

opt = parse(arg)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

if opt.GPU == -1 then
    nnlib = nn
else
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    if opt.nGPU == 1 then
        cutorch.setDevice(opt.GPU)
    end
end

if opt.branch ~= 'none' or opt.continue then
    -- Continuing training from a prior experiment
    -- Figure out which new options have been set
    local setOpts = {}
    for i = 1,#arg do
        if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
    end

    -- Where to load the previous options/model from
    if opt.branch ~= 'none' then opt.load = opt.expDir .. '/' .. opt.branch
    else opt.load = opt.expDir .. '/' .. opt.expID end

    -- Keep previous options, except those that were manually set
    local opt_ = opt
    opt = torch.load(opt_.load .. '/options.t7')
    opt.save = opt_.save
    opt.load = opt_.load
    opt.continue = opt_.continue
    for i = 1,#setOpts do opt[setOpts[i]] = opt_[setOpts[i]] end

    epoch = opt.lastEpoch + 1
    
    -- If there's a previous optimState, load that too
    if paths.filep(opt.load .. '/optimState.t7') then
        optimState = torch.load(opt.load .. '/optimState.t7')
        optimState.learningRate = opt.LR
    end

else epoch = 1 end
opt.epochNumber = epoch

-- Track accuracy
opt.acc = {train={}, valid={}}

-- Save options to experiment directory
torch.save(opt.save .. '/options.t7', opt)

end
