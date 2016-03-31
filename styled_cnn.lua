require('nn')
require('nngraph')
require('loadcaffe')
require('xlua')
require('image')
require('optim')
require('cunn')
require('cudnn')

require('./lib/tvloss')
require('./lib/contentloss')
require('./lib/gramloss')
require('./lib/mrfloss')
require('./lib/masked_gramloss')
require('./lib/amplayer')
require('./lib/randlayer')


local cleanupModel = require('./lib/cleanup_model')
local caffeImage = require('./lib/caffe_image')
local g = {}
g.styleImage = './images/picasso.png' 

g.trainImages_Path = './scene/'
g.trainImages_Number = 16657


-----------------------------------------------------------------------------------------
-- helper functions

string.startsWith = function(self, str) 
    return self:find('^' .. str) ~= nil
end

function loadVGG()
    local proto = './cnn/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt'
    local caffeModel = './cnn/vgg19/VGG_ILSVRC_19_layers.caffemodel'

    local fullModel = loadcaffe.load(proto, caffeModel, 'nn')
    local cnn = nn.Sequential()
    for i = 1, #fullModel do
        local name = fullModel:get(i).name
        if ( name:startsWith('relu') or name:startsWith('conv') or name:startsWith('pool') ) then
            cnn:add( fullModel:get(i) )
        else
            break
        end
    end

    fullModel = nil
    collectgarbage()
    return cnn
end

function loadTrainData() 
    local randSeq = torch.randperm(g.trainImages_Number)
    
    local trainSplit = math.floor(g.trainImages_Number * 0.85)

    g.trainSet = {}
    g.trainSet.data = {}
    g.trainSet.index = 1
    for i = 1, trainSplit do
        g.trainSet.data[i] = g.trainImages_Path .. '/' .. randSeq[i] .. '.png'
    end

    g.testSet = {}
    g.testSet.data = {}
    g.testSet.index = 1
    for i = trainSplit + 1, g.trainImages_Number do
        g.testSet.data[i] = g.trainImages_Path .. '/' .. randSeq[i] .. '.png'
    end
end

function loadBatch(set, batch_size) 
   
    local batch = {}
    batch.x = torch.Tensor(batch_size, 3, 256, 256)
    
    for i = 1, batch_size do
        local sampleIndex = i + set.index
        sampleIndex = sampleIndex % #set.data + 1

        local rgb = image.loadPNG( set.data[sampleIndex], 3)
        batch.x[i]:copy( caffeImage.img2caffe(rgb) )
    end

    set.index = (set.index + batch_size) % #set.data + 1

    return batch
end

-----------------------------------------------------------------------------------------
-- worker functions
function buildLossNet () 
    local gramLoss = {'relu1_2', 'relu2_2', 'relu3_2', 'relu4_1'}
    local contentLoss = {'relu4_2'}
    
    local styleCaffeImage = caffeImage.img2caffe( image.loadPNG(g.styleImage, 3) )   

    local modifier = {}
    local cindex = -1

    local net = nn.Sequential()
    net:add(nn.TVLoss(0.001))
   
    local gram_index = 1
    local content_index = 1
    for i = 1, #g.vgg do
        if ( gram_index > #gramLoss and content_index > #contentLoss) then
            break
        end
        
        local name = g.vgg:get(i).name
        net:add(g.vgg:get(i))
        
        if ( name == gramLoss[ gram_index ] ) then
            local target = net:forward( styleCaffeImage )
            local layer =  nn.GramLoss(0.0001, target, false)
            net:add(layer)
            table.insert(modifier, layer)

            gram_index = gram_index + 1
        end

        if ( name == contentLoss[content_index] ) then
            local layer =  nn.ContentLoss(1.0, nil, nil)
            net:add(layer)
            table.insert(modifier, layer)
            
            cindex = #modifier
            content_index = content_index + 1
        end
    end

    local lossNet = {}
    lossNet.net = net
    lossNet.modifier = modifier
    lossNet.cindex = cindex
    return lossNet
end

function buildStyledNet()
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.LeakyReLU(0.1))
    model:add(cudnn.SpatialConvolution(128, 3, 3, 3, 1, 1, 1, 1))
    model:add(nn.Tanh())
    model:add(nn.MulConstant(128))

    return model
end

function doTrain()
    g.lossNet.net:cuda()
    g.styledNet:cuda()
    g.zeroLoss = g.zeroLoss:cuda()    

    g.styledNet:training()
    g.lossNet.net:evaluate()

    local batchSize = 4
    local oneEpoch = math.floor( #g.trainSet.data / batchSize ) 
    g.trainSet.index = 1
   
    local batch = nil
    local dyhat = torch.zeros(batchSize, 3, 256, 256):cuda()
    local parameters,gradParameters = g.styledNet:getParameters() 
    
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end
        -- reset gradients
        gradParameters:zero()
        
        local loss = 0
        local yhat = g.styledNet:forward( batch.x )
        
        for i = 1, batchSize do
            g.lossNet.net:forward( batch.x[i] )
            local contentTarget = g.lossNet.modifier[g.lossNet.cindex].output
            g.lossNet.modifier[g.lossNet.cindex]:setTarget(contentTarget)

            g.lossNet.net:forward(yhat[i])
            local dy = g.lossNet.net:backward(yhat[i], g.zeroLoss)
            dyhat[i]:copy(dy) 

            for _, mod in ipairs(g.lossNet.modifier) do
                loss = loss + mod.loss
            end
        end

        g.styledNet:backward(batch.x, dyhat) 
    
        return loss/batchSize, gradParameters
    end
    
    local minValue = -1
    for j = 1, oneEpoch do
        batch = loadBatch(g.trainSet, batchSize)
        batch.x = batch.x:cuda()

        local _, err = optim.adam(feval, parameters, g.optimState)

        print(">>>>>>>>> err = " .. err[1]);
        
        if ( j % 1000 == 0) then
            local model = g.styledNet:clone()
            cleanupModel(model)
            torch.save('./model/style_' .. err[1] .. '.t7',  model)
        end

        collectgarbage();
    end
    
end

function doTest() 


end


function doForward() 
    local net = torch.load( arg[1] )
    local img = image.loadPNG( arg[2] , 3)

    local img = caffeImage.img2caffe(img)
    img = img:cuda()

    local outImg = net:forward(img)
    outImg = outImg:float()
    outImg = caffeImage.caffe2img(outImg)

    image.savePNG('./output.png', outImg)

end


-----------------------------------------------------------------------------------------
function main() 
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1979)

    if ( #arg == 2) then
        doForward()
        return
    end
    
    
    -- build net
    g.vgg = loadVGG()
    g.lossNet = buildLossNet()
    local tempImage = torch.rand(3, 256, 256)
    local tempOutput = g.lossNet.net:forward(tempImage)
    g.zeroLoss = torch.zeros( tempOutput:size())

    g.styledNet = buildStyledNet()
    g.optimState = {
        learningRate = 0.0001,
    }

    -- load data
    loadTrainData()

    -- trainging()
    for i = 1, 4 do
        doTrain()
        doTest()
    end
end

main()
