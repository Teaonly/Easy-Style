require('nn')

local RandLayer, parent = torch.class('nn.RandLayer', 'nn.Module')

function RandLayer:__init()
    parent.__init(self)
    self.randMap = nil
    self.loss = 0
end

function RandLayer:updateOutput(input)
    self.output = input
    
    if ( self.randMap  == nil or self.randMap:isSameSizeAs(input)) then
        self.randMap = torch.rand(input:size()) * -1
    end

    return self.output
end

function RandLayer:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):copy(input)

    self.gradInput:cmul(self.randMap)
    self.gradInput:add(gradOutput)
    return self.gradInput
end

