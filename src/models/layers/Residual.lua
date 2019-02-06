-- Defines the basic residual block used in the network specification


-- Spatial convolution layer from the 'nn' module
local conv = nnlib.SpatialConvolution
-- Spatial BatchNorm layer from the 'nn' module
local batchnorm = nn.SpatialBatchNormalization
-- ReLU layer
local relu = nnlib.ReLU


-- Main convolutional block
-- Inputs
-- numIn: number of input channels
-- numOut: number of output channels
local function convBlock(numIn, numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end


-- Skip layer
-- Inputs
-- numIn: number of input channels
-- numOut: number of output channels
local function skipLayer(numIn, numOut)
    -- If the number of output channels is same as that of the input, an identity layer can be returned
    if numIn == numOut then
        return nn.Identity()
    -- Else, 1 x 1 convolutions can be applied to get the input to the desired dimensionality
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end


-- Residual block
-- Inputs
-- numIn: number of input channels
-- numOut: number of output channels
function Residual(numIn, numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

