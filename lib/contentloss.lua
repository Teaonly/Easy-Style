require('nn')

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
    parent.__init(self)
    self.strength = strength
    if ( target ~= nil) then
        self.target = target:clone()
    end
    self.normalize = normalize or false
    self.loss = 0
    self.crit = nn.MSECriterion()
end

function ContentLoss:setTarget(target)
    self.target = target:clone()
end

function ContentLoss:updateOutput(input)
    if self.target and input:nElement() == self.target:nElement() then
        self.loss = self.crit:forward(input, self.target) * self.strength 
    end
    
    self.output = input
    return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
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


