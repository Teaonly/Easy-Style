require('nn')

local buildMRFTarget = function(input, ref)
    -- this is a simple version , no scale, no rotation
    local channel = input:size()[1]
    local height = input:size()[2]
    local width = input:size()[3]
    local width_ref = ref:size()[3]
 
    local normConv = nn.SpatialConvolutionMM(channel, 1, 3, 3, 1, 1, 0, 0)
    normConv.weight:fill(1.0)
    normConv.bias:fill(0)
    local normValue = normConv:forward(ref:abs())
    normValue:div(9*channel)

    local target = input:clone()

   local x, y =  1,  1
    while true do
        -- processing patch by patch
        if ( y > height - 2) then
            break
        end
        
        local conv = nn.SpatialConvolution(channel, 1, 3, 3, 1, 1, 0, 0) 
        conv.weight[1]:copy( input[{{}, {y,y+2}, {x, x+2}}] )
        conv.bias:fill(0)
        
        local scores = conv:forward(ref)
        scores:cdiv(normValue)

        -- find best match patch from reference images
        local _, pos = scores:view(-1):max(1)
        pos = pos[1] - 1
        local bestX = pos % ( width_ref - 2) + 1 
        local bestY = math.floor( pos / ( width_ref - 2) ) + 1
        
        target[{{}, {y,y+2}, {x, x+2}}]:copy( ref[{{},{bestY, bestY+2},{bestX, bestX+2}}])

        x = x + 3
        if ( x > width - 2) then
            x = 1
            y = y + 3
        end
        collectgarbage()
    end
    
    return target
end


local MRFLoss, parent = torch.class('nn.MRFLoss', 'nn.Module')

function MRFLoss:__init(strength, input, ref, normalize)
    parent.__init(self)
    self.normalize = normalize or false
    self.strength = strength
    self.loss = 0
    self.crit = nn.MSECriterion()
    
    self.target = buildMRFTarget(input, ref)
end


function MRFLoss:updateOutput(input)
    if ( self.target:nElement() == input:nElement() ) then
        self.loss = self.crit:forward(input, self.target) * self.strength
    end
   
    self.output = input
    return self.output
end

function MRFLoss:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()

    if input:nElement() == self.target:nElement() then
        self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
        self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
end




