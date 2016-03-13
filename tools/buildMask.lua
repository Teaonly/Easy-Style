-- stupid tools only support: red, green, blue and white 4 colors masks
--

require('image')

function main()
    if ( #arg ~= 3) then
        print("Please input [style_mask_png_file], [target_mask_png_file], [output_mask_file] ")
        print("Only support red, green, blue and white 4 colors masks")
        return
    end

    local style_mask = image.load(arg[1], 3)
    local target_mask = image.load(arg[2], 3)

    if (    style_mask:size()[2] ~= target_mask:size()[2]
         or style_mask:size()[3] ~= target_mask:size()[3] ) then
        print("Error: style_mask and target_mask must be same size")
        return
    end

    local width = style_mask:size()[3]
    local height = style_mask:size()[2]

    -- only support 4 channels
    local masks = {}  
    masks.style = torch.zeros(4, height, width)
    masks.target = torch.zeros(4, height, width)

    -- white only
    local whiteMask = torch.zeros(height, width):byte()
    for i = 1, 3 do
        whiteMask:add(style_mask[i]:le(0.5) * (-1) + 1)
    end
    whiteMask = whiteMask:le(2.5) * (-1) + 1
    masks.style[1] = whiteMask:clone()
    whiteMask:zero()
    for i = 1, 3 do
        whiteMask:add(target_mask[i]:le(0.5) * (-1) + 1)
    end
    whiteMask = whiteMask:le(2.5) * (-1) + 1
    masks.target[1] = whiteMask:clone()
    
    -- Red, Green, Blue
    for i = 1, 3 do
        local j = (i + 1) % 3 + 1

        local colorMask = (target_mask[i]:le(0.5) * (-1) + 1)
        colorMask:cmul( target_mask[j]:le(0.5) )
        masks.target[i+1] = colorMask:clone()
 
        colorMask = (style_mask[i]:le(0.5) * (-1) + 1)
        colorMask:cmul( style_mask[j]:le(0.5) )
        masks.style[i+1] = colorMask:clone()
    end

    torch.save(arg[3], masks)
end

main()
