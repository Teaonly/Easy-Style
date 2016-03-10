require('nn')

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local GramLoss, parent = torch.class('nn.GramLoss', 'nn.Module')

function GramLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
  
  local tsize = target:size()
  local img_size = tsize[2] * tsize[3]
  
  self.target = self.gram:forward(target):clone()
  self.target:div( img_size ) 
end

function GramLoss:updateOutput(input)
  local tsize = input:size()
  local img_size = tsize[2] * tsize[3]

  self.G = self.gram:forward(input)
  self.G:div(img_size)
  
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function GramLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)

  local tsize = input:size()
  local img_size = tsize[2] * tsize[3]
  dG:mul(img_size)
  
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

