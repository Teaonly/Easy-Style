require('nn')
require('nngraph')
require('loadcaffe')
require('xlua')
require('image')
require('optim')
require('cunn')

require('./lib/tvloss')
require('./lib/contentloss')
require('./lib/gramloss')
require('./lib/mrfloss')
require('./lib/masked_gramloss')
require('./lib/amplayer')
require('./lib/randlayer')

local caffeImage = require('./lib/caffe_image')

----------------------------------------------------------------------------------------
local g = {}

local doRevert = function()
    local currentImage = g.x
    for i = 1, g.conf.maxIterate do
        local inout = g.net:forward(currentImage)
        inout = g.net:backward(currentImage, g.dy)
        
        currentImage:add(inout * (-1 * g.conf.step) );
        currentImage:clamp(-128,128)

        collectgarbage()
        xlua.progress(i, g.conf.maxIterate)
    end
end

local doConvergence = function()
    local optim_state = {
        maxIter = g.conf.maxIterate,
        verbose = true,
    }

    local num_calls = 0
    local function feval(x)

        num_calls = num_calls + 1

        g.net:forward(x)
        local grad = g.net:updateGradInput(x, g.dy)
        local loss = 0
        
        for _, mod in ipairs(g.modifier) do
            loss = loss + mod.loss
        end
        
        print(">>>>>>>>>" .. loss)
        --xlua.progress(num_calls, optim_state.maxIter)

        collectgarbage()
        -- optim.lbfgs expects a vector for gradients
        return loss, grad:view(grad:nElement())
    end
    
    local x, losses = optim.lbfgs(feval, g.x, optim_state)  
end

local main = function()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1979)
    if ( #arg < 1) then
        print("Please input config file!")
        os.exit(0)
    end

    -- init 
    g.conf = dofile(arg[1])
    g.cnn = loadCNN(g.conf.cnn)
    g.net, g.modifier = buildNetwork(g.conf, g.cnn) 
    g.x = loadInput(g.conf)
    g.dy = torch.zeros( g.net:forward(g.x):size() )

    -- cuda 
    g.net:cuda()
    g.x = g.x:cuda()
    g.dy = g.dy:cuda()

    print(g.net)

    collectgarbage()
    if (g.conf.convergence) then
        doConvergence()                     
    else
        doRevert()
    end
    
    local img = caffeImage.caffe2img(g.x:float())
    image.savePNG(g.conf.image_list[g.conf.output], img)
end

-----------------------------------------------------------------------------------------
-- helper functions

string.startsWith = function(self, str) 
    return self:find('^' .. str) ~= nil
end

function loadInput(conf) 
    local img = nil
    if ( conf.image_list[conf.input] == nil ) then
        img = torch.rand(3, 256, 256) * 0.01
    else
        img = image.load(conf.image_list[conf.input], 3)
    end

    img = caffeImage.img2caffe(img)
    return img
end

function loadCNN(cnnFiles)
    local fullModel = loadcaffe.load(cnnFiles.proto, cnnFiles.caffemodel, 'nn')
    local cnn = nn.Sequential();
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

function buildNetwork(conf, cnn) 
    local net = nn.Sequential()
    local modifier = {}

    local nindex = 1
    if ( conf.net[1].layer == 'input') then
        local layer = buildLayer(net, conf, 1, cnn)
        net:add(layer)
        nindex = 2
    end

    for i = 1, #cnn do
        local name = cnn:get(i).name
        net:add(cnn:get(i))
        
        if ( name == conf.net[nindex].layer ) then
            local layer = buildLayer(net, conf, nindex, cnn)
            net:add(layer)
            
            table.insert(modifier, layer)
                
            nindex = nindex + 1
            if ( nindex > #conf.net ) then
                break 
            end
        end
        collectgarbage()
    end

    return net, modifier
end

function buildLayer(net, conf, nindex, cnn) 
    local layer = nil
    
    if ( conf.net[nindex].type == "tvloss" ) then
        layer = nn.TVLoss(conf.net[nindex].weight)
    elseif ( conf.net[nindex].type == "amp") then
        layer = nn.AmpLayer(conf.net[nindex].ratio)  
    elseif ( conf.net[nindex].type == "rand") then
        layer = nn.RandLayer()
    elseif ( conf.net[nindex].type == "content") then
        local targetImage = conf.image_list[ conf.net[nindex].target]
        targetImage = image.load(targetImage,3)
        local targetCaffe = caffeImage.img2caffe(targetImage)
        local target = net:forward(targetCaffe)
            
        layer = nn.ContentLoss(conf.net[nindex].weight, target)
    elseif ( conf.net[nindex].type == "gram") then
        local targetImage = conf.image_list[ conf.net[nindex].target]
        targetImage = image.load(targetImage,3)
        local targetCaffe = caffeImage.img2caffe(targetImage)
        local target = net:forward(targetCaffe)
    
        layer = nn.GramLoss(conf.net[nindex].weight, target)
    elseif ( conf.net[nindex].type == "mrf") then
        local targetImage = conf.image_list[ conf.net[nindex].target]
        targetImage = image.load(targetImage,3)
        local targetCaffe = caffeImage.img2caffe(targetImage)
        local target = net:forward(targetCaffe):clone()
   
        local inputImage = conf.image_list[ conf.input]
        inputImage = image.load(inputImage, 3)
        local inputCaffe = caffeImage.img2caffe(inputImage)
        local input = net:forward(inputCaffe)

        layer = nn.MRFLoss(conf.net[nindex].weight, input, target)
    elseif ( conf.net[nindex].type == 'mask_gram') then
        local styleImage = conf.image_list[ conf.net[nindex].style] 
        styleImage = image.load(styleImage,3)
        local styleCaffe = caffeImage.img2caffe(styleImage)
        local style = net:forward(styleCaffe):clone() 

        local masks = torch.load ( conf.image_list[ conf.net[nindex].mask], 'ascii')

        layer = nn.MaskedGramLoss(conf.net[nindex].weight, style, masks)
    else
        print(">>>>>>>>>>>>>>>")
    end
    
    return layer
end

-----------------------------------------------------------------------------------------
main()

