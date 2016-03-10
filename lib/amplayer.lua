require('nn')

local AmpLayer, parent = torch.class('nn.AmpLayer', 'nn.Module')

function AmpLayer:__init(ratio)
    parent.__init(self)
    self.ratio = ratio
    self.loss = 0
end

function AmpLayer:updateOutput(input)
    self.output = input
    return self.output
end

function AmpLayer:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):copy(input)

    self.gradInput:mul(-1*self.ratio)
    self.gradInput:add(gradOutput)
    return self.gradInput
end

