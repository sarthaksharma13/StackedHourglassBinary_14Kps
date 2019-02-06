-- Script to load a hg-unary model (pretrained), and copy its weights to a hg-binary model
-- We make use of a 'dummy' hg-binary snapshot, i.e., we load a dummy hg-binary snapshot, 
-- copy the trained hg-unary weights (wherever relevant), and save the new hg-binary snapshot


-- Load requried packages
require 'torch'
require 'paths'
require 'io'
require 'string'
require 'image'

require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'

print('Loaded all torch packages ...')


-- Function to copy weights of one residual module to another
-- Inputs:
-- resBinary: reference to residual block of the binary net
-- resUnary: reference to residual block of the unary net
function copyResidualBlock(resBinary, resUnary)

    -- First BN layer
    resBinary:get(1):get(1):get(1).weight = resUnary:get(1):get(1):get(1).weight
    resBinary:get(1):get(1):get(1).bias = resUnary:get(1):get(1):get(1).bias
    resBinary:get(1):get(1):get(1).gradWeight = resUnary:get(1):get(1):get(1).gradWeight
    resBinary:get(1):get(1):get(1).gradBias = resUnary:get(1):get(1):get(1).gradBias
    -- Conv (N -> N/2)
    resBinary:get(1):get(1):get(3).weight = resUnary:get(1):get(1):get(3).weight
    resBinary:get(1):get(1):get(3).bias = resUnary:get(1):get(1):get(3).bias
    resBinary:get(1):get(1):get(3).gradWeight = resUnary:get(1):get(1):get(3).gradWeight
    resBinary:get(1):get(1):get(3).gradBias = resUnary:get(1):get(1):get(3).gradBias
    -- BN
    resBinary:get(1):get(1):get(4).weight = resUnary:get(1):get(1):get(4).weight
    resBinary:get(1):get(1):get(4).bias = resUnary:get(1):get(1):get(4).bias
    resBinary:get(1):get(1):get(4).gradWeight = resUnary:get(1):get(1):get(4).gradWeight
    resBinary:get(1):get(1):get(4).gradBias = resUnary:get(1):get(1):get(4).gradBias
    -- Conv (N/2 -> N/2)
    resBinary:get(1):get(1):get(6).weight = resUnary:get(1):get(1):get(6).weight
    resBinary:get(1):get(1):get(6).bias = resUnary:get(1):get(1):get(6).bias
    resBinary:get(1):get(1):get(6).gradWeight = resUnary:get(1):get(1):get(6).gradWeight
    resBinary:get(1):get(1):get(6).gradBias = resUnary:get(1):get(1):get(6).gradBias
    -- BN
    resBinary:get(1):get(1):get(7).weight = resUnary:get(1):get(1):get(7).weight
    resBinary:get(1):get(1):get(7).bias = resUnary:get(1):get(1):get(7).bias
    resBinary:get(1):get(1):get(7).gradWeight = resUnary:get(1):get(1):get(7).gradWeight
    resBinary:get(1):get(1):get(7).gradBias = resUnary:get(1):get(1):get(7).gradBias
    -- Conv (N/2 -> N)
    resBinary:get(1):get(1):get(9).weight = resUnary:get(1):get(1):get(9).weight
    resBinary:get(1):get(1):get(9).bias = resUnary:get(1):get(1):get(9).bias
    resBinary:get(1):get(1):get(9).gradWeight = resUnary:get(1):get(1):get(9).gradWeight
    resBinary:get(1):get(1):get(9).gradBias = resUnary:get(1):get(1):get(9).gradBias

end


-- Function to copy weights of one hourglass to the other
-- Note that this is for one hourglass, not the entire stack
-- Inputs:
-- hgBinary: reference to the hg-binary net
-- hgUnary: reference to the hg-unary net
-- hgBinaryStartIdx: start index of hourglass module of the binary net
-- hgUnaryInds: start index of hourglass module of the unary net
function copyHourglass(hgBinary, hgUnary, hgBinaryStartIdx, hgUnaryStartIdx)

    -- Downsampling stage
    -- First residual block
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx), hgUnary:get(hgUnaryStartIdx))
    -- Second residual block + branch
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+2), hgUnary:get(hgUnaryStartIdx+2))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+3), hgUnary:get(hgUnaryStartIdx+3))
    -- Third residual block + branch
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+5), hgUnary:get(hgUnaryStartIdx+5))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+6), hgUnary:get(hgUnaryStartIdx+6))
    -- Fourth residual block + branch
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+8), hgUnary:get(hgUnaryStartIdx+8))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+9), hgUnary:get(hgUnaryStartIdx+9))


    -- Middle of the hourglass
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+11), hgUnary:get(hgUnaryStartIdx+11))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+12), hgUnary:get(hgUnaryStartIdx+12))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+13), hgUnary:get(hgUnaryStartIdx+13))


    -- Upsampling stage
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+16), hgUnary:get(hgUnaryStartIdx+16))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+19), hgUnary:get(hgUnaryStartIdx+19))
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx+22), hgUnary:get(hgUnaryStartIdx+22))

end


-- Function to copy the weights post the hourglass from the unary net to the binary net
-- Note that this has to be called once for each hourglass
-- Inputs:
-- hgBinary: reference to the binary net
-- hgUnary: reference to the unary net
-- hgBinaryStartIdx: start idx of the module after the hourglass in the binary net
-- hgUnaryStartIdx: start idx of the module after the hourglass in the unary net
function copyPostHourglass(hgBinary, hgUnary, hgBinaryStartIdx, hgUnaryStartIdx)

    -- Res (N -> N)
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx), hgUnary:get(hgUnaryStartIdx))
    -- Conv (256 -> 256)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+1, hgUnaryStartIdx+1)
    -- hgBinary:get(hgBinaryStartIdx+1).weight = hgUnary:get(hgUnaryStartIdx+1).weight
    -- hgBinary:get(hgBinaryStartIdx+1).bias = hgUnary:get(hgUnaryStartIdx+1).bias
    -- hgBinary:get(hgBinaryStartIdx+1).gradWeight = hgUnary:get(hgUnaryStartIdx+1).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+1).gradBias = hgUnary:get(hgUnaryStartIdx+1).gradBias
    -- BN
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+2, hgUnaryStartIdx+2)
    -- hgBinary:get(hgBinaryStartIdx+2).weight = hgUnary:get(hgUnaryStartIdx+2).weight
    -- hgBinary:get(hgBinaryStartIdx+2).bias = hgUnary:get(hgUnaryStartIdx+2).bias
    -- hgBinary:get(hgBinaryStartIdx+2).gradWeight = hgUnary:get(hgUnaryStartIdx+2).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+2).gradBias = hgUnary:get(hgUnaryStartIdx+2).gradBias
    -- Conv (256 -> 14)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+4, hgUnaryStartIdx+4)
    -- hgBinary:get(hgBinaryStartIdx+4).weight = hgUnary:get(hgUnaryStartIdx+4).weight
    -- hgBinary:get(hgBinaryStartIdx+4).bias = hgUnary:get(hgUnaryStartIdx+4).bias
    -- hgBinary:get(hgBinaryStartIdx+4).gradWeight = hgUnary:get(hgUnaryStartIdx+4).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+4).gradBias = hgUnary:get(hgUnaryStartIdx+4).gradBias
    -- Conv (256 -> 256)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+9, hgUnaryStartIdx+5)
    -- hgBinary:get(hgBinaryStartIdx+9).weight = hgUnary:get(hgUnaryStartIdx+5).weight
    -- hgBinary:get(hgBinaryStartIdx+9).bias = hgUnary:get(hgUnaryStartIdx+5).bias
    -- hgBinary:get(hgBinaryStartIdx+9).gradWeight = hgUnary:get(hgUnaryStartIdx+5).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+9).gradBias = hgUnary:get(hgUnaryStartIdx+5).gradBias
    -- Conv (14 -> 256)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+10, hgUnaryStartIdx+6)
    -- hgBinary:get(hgBinaryStartIdx+10).weight = hgUnary:get(hgUnaryStartIdx+6).weight
    -- hgBinary:get(hgBinaryStartIdx+10).bias = hgUnary:get(hgUnaryStartIdx+6).bias
    -- hgBinary:get(hgBinaryStartIdx+10).gradWeight = hgUnary:get(hgUnaryStartIdx+6).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+10).gradBias = hgUnary:get(hgUnaryStartIdx+6).gradBias
    -- Conv (14 -> 256)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+11, hgUnaryStartIdx+6)
    -- hgBinary:get(hgBinaryStartIdx+11).weight = hgUnary:get(hgUnaryStartIdx+6).weight
    -- hgBinary:get(hgBinaryStartIdx+11).bias = hgUnary:get(hgUnaryStartIdx+6).bias
    -- hgBinary:get(hgBinaryStartIdx+11).gradWeight = hgUnary:get(hgUnaryStartIdx+6).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+11).gradBias = hgUnary:get(hgUnaryStartIdx+6).gradBias

end


-- Copy the parts after the end of the last hourglass
function copyAfterLastHourglass(hgBinary, hgUnary, hgBinaryStartIdx, hgUnaryStartIdx)

    -- Res (N -> N)
    copyResidualBlock(hgBinary:get(hgBinaryStartIdx), hgUnary:get(hgUnaryStartIdx))
    -- Conv (256 -> 256)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+1, hgUnaryStartIdx+1)
    -- hgBinary:get(hgBinaryStartIdx+1).weight = hgUnary:get(hgUnaryStartIdx+1).weight
    -- hgBinary:get(hgBinaryStartIdx+1).bias = hgUnary:get(hgUnaryStartIdx+1).bias
    -- hgBinary:get(hgBinaryStartIdx+1).gradWeight = hgUnary:get(hgUnaryStartIdx+1).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+1).gradBias = hgUnary:get(hgUnaryStartIdx+1).gradBias
    -- BN
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+2, hgUnaryStartIdx+2)
    -- hgBinary:get(hgBinaryStartIdx+2).weight = hgUnary:get(hgUnaryStartIdx+2).weight
    -- hgBinary:get(hgBinaryStartIdx+2).bias = hgUnary:get(hgUnaryStartIdx+2).bias
    -- hgBinary:get(hgBinaryStartIdx+2).gradWeight = hgUnary:get(hgUnaryStartIdx+2).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+2).gradBias = hgUnary:get(hgUnaryStartIdx+2).gradBias
    -- Conv (256 -> 14)
    copyModuleWeights(hgBinary, hgUnary, hgBinaryStartIdx+4, hgUnaryStartIdx+4)
    -- hgBinary:get(hgBinaryStartIdx+4).weight = hgUnary:get(hgUnaryStartIdx+4).weight
    -- hgBinary:get(hgBinaryStartIdx+4).bias = hgUnary:get(hgUnaryStartIdx+4).bias
    -- hgBinary:get(hgBinaryStartIdx+4).gradWeight = hgUnary:get(hgUnaryStartIdx+4).gradWeight
    -- hgBinary:get(hgBinaryStartIdx+4).gradBias = hgUnary:get(hgUnaryStartIdx+4).gradBias

end


-- Function to copy a single module's weight, bias, gradWeight, gradBias
function copyModuleWeights(hgBinary, hgUnary, hgBinaryIdx, hgUnaryIdx)

    -- Check dimensions for weights and copy them if consistent
    if #hgBinary:get(hgBinaryIdx).weight:size() == #hgUnary:get(hgUnaryIdx).weight:size() then
        isSameSize = true
        for k = 1, #hgBinary:get(hgBinaryIdx).weight:size() do
            if hgBinary:get(hgBinaryIdx).weight:size(k) ~= hgUnary:get(hgUnaryIdx).weight:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent weight sizes')
        else
            hgBinary:get(hgBinaryIdx).weight = hgUnary:get(hgUnaryIdx).weight:clone()
        end
    else
        print('Inconsistent number of dimensions for weight')
    end

    -- Check dimensions for bias and copy them if consistent
    if #hgBinary:get(hgBinaryIdx).bias:size() == #hgUnary:get(hgUnaryIdx).bias:size() then
        isSameSize = true
        for k = 1, #hgBinary:get(hgBinaryIdx).bias:size() do
            if hgBinary:get(hgBinaryIdx).bias:size(k) ~= hgUnary:get(hgUnaryIdx).bias:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent bias sizes')
        else
            hgBinary:get(hgBinaryIdx).bias = hgUnary:get(hgUnaryIdx).bias:clone()
        end
    else
        print('Inconsistent number of dimensions for bias')
    end

    -- Check dimensions for gradWeight and copy them if consistent
    if #hgBinary:get(hgBinaryIdx).gradWeight:size() == #hgUnary:get(hgUnaryIdx).gradWeight:size() then
        isSameSize = true
        for k = 1, #hgBinary:get(hgBinaryIdx).gradWeight:size() do
            if hgBinary:get(hgBinaryIdx).gradWeight:size(k) ~= hgUnary:get(hgUnaryIdx).gradWeight:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent gradWeight sizes')
        else
            hgBinary:get(hgBinaryIdx).gradWeight = hgUnary:get(hgUnaryIdx).gradWeight:clone()
        end
    else
        print('Inconsistent number of dimensions for gradWeight')
    end

    -- Check dimensions for bias and copy them if consistent
    if #hgBinary:get(hgBinaryIdx).gradBias:size() == #hgUnary:get(hgUnaryIdx).gradBias:size() then
        isSameSize = true
        for k = 1, #hgBinary:get(hgBinaryIdx).gradBias:size() do
            if hgBinary:get(hgBinaryIdx).gradBias:size(k) ~= hgUnary:get(hgUnaryIdx).gradBias:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent gradBias sizes')
        else
            hgBinary:get(hgBinaryIdx).gradBias = hgUnary:get(hgUnaryIdx).gradBias:clone()
        end
    else
        print('Inconsistent number of dimensions for gradBias')
    end

end


function copyResidualBlockHelper(moduleBinary, moduleUnary, moduleBinaryIdx, moduleUnaryIdx)

    -- Check dimensions for weights and copy them if consistent
    if #moduleBinary:get(1):get(1):get(moduleBinaryIdx).weight:size() == #moduleUnary:get(1):get(1):get(moduleUnaryIdx).weight:size() then
        isSameSize = true
        for k = 1, #moduleBinary:get(1):get(1):get(moduleBinaryIdx).weight:size() do
            if moduleBinary:get(1):get(1):get(moduleBinaryIdx).weight:size(k) ~= moduleUnary:get(1):get(1):get(moduleUnaryIdx).weight:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent weight sizes in residual block component')
        else
            moduleBinary:get(1):get(1):get(moduleBinaryIdx).weight = moduleUnary:get(1):get(1):get(moduleUnaryIdx).weight:clone()
        end
    else
        print('Inconsistent number of dimensions for weight in residual block component')
    end

    -- Check dimensions for bias and copy them if consistent
    if #moduleBinary:get(1):get(1):get(moduleBinaryIdx).bias:size() == #moduleUnary:get(1):get(1):get(moduleUnaryIdx).bias:size() then
        isSameSize = true
        for k = 1, #moduleBinary:get(1):get(1):get(moduleBinaryIdx).bias:size() do
            if moduleBinary:get(1):get(1):get(moduleBinaryIdx).bias:size(k) ~= moduleUnary:get(1):get(1):get(moduleUnaryIdx).bias:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent bias sizes in residual block component')
        else
            moduleBinary:get(1):get(1):get(moduleBinaryIdx).bias = moduleUnary:get(1):get(1):get(moduleUnaryIdx).bias:clone()
        end
    else
        print('Inconsistent number of dimensions for bias in residual block component')
    end

    -- Check dimensions for gradWeight and copy them if consistent
    if #moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradWeight:size() == #moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradWeight:size() then
        isSameSize = true
        for k = 1, #moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradWeight:size() do
            if moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradWeight:size(k) ~= moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradWeight:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent gradWeight sizes in residual block component')
        else
            moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradWeight = moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradWeight:clone()
        end
    else
        print('Inconsistent number of dimensions for gradWeight in residual block component')
    end

    -- Check dimensions for gradBias and copy them if consistent
    if #moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradBias:size() == #moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradBias:size() then
        isSameSize = true
        for k = 1, #moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradBias:size() do
            if moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradBias:size(k) ~= moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradBias:size(k) then
                isSameSize = false
            end
        end
        if isSameSize == false then
            print('Inconsistent gradBias sizes in residual block component')
        else
            moduleBinary:get(1):get(1):get(moduleBinaryIdx).gradBias = moduleUnary:get(1):get(1):get(moduleUnaryIdx).gradBias:clone()
        end
    else
        print('Inconsistent number of dimensions for gradBias in residual block component')
    end

end


-- Path to the unary and binary models
unaryPath = '/home/km/code/hourglass-binary/exp/pascal3D/hg-car-kps-pascal.t7'
binaryPath = '/home/km/code/hourglass-binary/exp/pascal3D/hg-binary-poor.t7'

-- Load the models (deserialize) and remove the DataParallelTable, i.e., we require only gModules
print('Loading models and deserializing ...')
unaryModel = torch.load(unaryPath)
unaryModel = unaryModel:get(1)
binaryModel = torch.load(binaryPath)
binaryModel = binaryModel:get(1)


print('Copying weights ...')

print('Initial parts ...')

-- Conv (3 -> 64, 3 x 3, 1, 1, 1, 1)
copyModuleWeights(binaryModel, unaryModel, 2, 2)
-- binaryModel:get(2).bias = unaryModel:get(2).bias
-- binaryModel:get(2).gradWeight = unaryModel:get(2).gradWeight
-- binaryModel:get(2).gradBias = unaryModel:get(2).gradBias

-- BN
copyModuleWeights(binaryModel, unaryModel, 3, 3)
-- binaryModel:get(3).weight = unaryModel:get(3).weight
-- binaryModel:get(3).bias = unaryModel:get(3).bias
-- binaryModel:get(3).gradWeight = unaryModel:get(3).gradWeight
-- binaryModel:get(3).gradBias = unaryModel:get(3).gradBias
-- Residual blocks before the first hourglass
-- Res(64, 128)
copyResidualBlock(binaryModel:get(5), unaryModel:get(5))
-- Res(128, 128)
copyResidualBlock(binaryModel:get(6), unaryModel:get(6))
-- Res (128, 256)
copyResidualBlock(binaryModel:get(7), unaryModel:get(7))

print('Hourglasses ...')
copyHourglass(binaryModel, unaryModel, 8, 8)
copyHourglass(binaryModel, unaryModel, 46, 41)
copyHourglass(binaryModel, unaryModel, 84, 74)
copyHourglass(binaryModel, unaryModel, 122, 107)
copyHourglass(binaryModel, unaryModel, 160, 140)
copyHourglass(binaryModel, unaryModel, 198, 173)
copyHourglass(binaryModel, unaryModel, 236, 206)
copyHourglass(binaryModel, unaryModel, 274, 239)

print('Post hourglasses ...')
copyPostHourglass(binaryModel, unaryModel, 33, 33)
copyPostHourglass(binaryModel, unaryModel, 71, 66)
copyPostHourglass(binaryModel, unaryModel, 109, 99)
copyPostHourglass(binaryModel, unaryModel, 147, 132)
copyPostHourglass(binaryModel, unaryModel, 185, 165)
copyPostHourglass(binaryModel, unaryModel, 223, 198)
copyPostHourglass(binaryModel, unaryModel, 261, 231)

print('After last hourglass ...')
copyAfterLastHourglass(binaryModel, unaryModel, 299, 264)

print('Saving snapshot ...')
torch.save('/home/km/code/hourglass-binary/exp/pascal3D/hg-binary-copied.t7', binaryModel)
