require('nn')
require('image')

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
local MaskedGramLoss, parent = torch.class('nn.MaskedGramLoss', 'nn.Module')

function MaskedGramLoss:__init(strength, style, masks)
  parent.__init(self)
  self.strength = strength
  self.loss = 0
  self.crit = nn.MSECriterion()
    
  local channel = style:size()[1]
  local hei = style:size()[2]
  local wid = style:size()[3]

  local gram = GramMatrix()

  local allGramTarget = {}

  local maskedLoss = nn.ConcatTable()
  for i = 1, #masks.style do
    local style_mask = image.scale(masks.style[i], wid, hei):float()
    
    style_mask = style_mask:view(1, hei, wid):expandAs(style)
    style_mask = torch.cmul(style_mask, style)

    allGramTarget[i] = gram:forward(style_mask):clone()
    allGramTarget[i]:div(wid*hei)

    local target_mask = image.scale(masks.target[i], wid, hei):float()
    target_mask = target_mask:view(1, hei, wid):expandAs(style)

    local mask_net = nn.Sequential()
    local cmul = nn.CMul(style:size())
    cmul.weight:copy( target_mask)

    mask_net:add(cmul)
    mask_net:add(GramMatrix())
    maskedLoss:add(mask_net)
  end
  
  self.allGramTarget = allGramTarget
  self.maskedLoss = maskedLoss
 
end

function MaskedGramLoss:updateOutput(input)
  local tsize = input:size()
  local img_size = tsize[2] * tsize[3]
    
  self.loss = 0
  local maskedGram = self.maskedLoss:forward(input) 
  for i = 1, #maskedGram do
    maskedGram[i]:div(img_size)
    self.loss = self.loss + self.crit:forward(maskedGram[i], self.allGramTarget[i]) 
  end
  self.maskedGram = maskedGram

  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function MaskedGramLoss:updateGradInput(input, gradOutput)
  local tsize = input:size()
  local img_size = tsize[2] * tsize[3]
 
  local dG = {}
  for i = 1, #self.allGramTarget do
    dG[i] = self.crit:backward(self.maskedGram[i], self.allGramTarget[i]):clone()
    dG[i]:mul(img_size)
  end

  self.gradInput = self.maskedLoss:backward(input, dG)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

