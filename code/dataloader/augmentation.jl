struct RandomCrop
    width;
    height;
    depth;
end

function (c::RandomCrop)(img, mask) 
    width, height, depth = size(img)
    x = rand(1 : width - c.width)
    y = rand(1 : height - c.height)
    z = rand(1 : depth - c.depth)
    img = img[x : x + c.width-1, y : y + c.height-1, z : z + c.depth-1]
    mask = mask[x : x + c.width-1, y : y + c.height-1, z : z + c.depth-1]
    return img, mask
end

#=
Augmentor: Flip + rot90 + randomcrop
=#

struct Augmentor
    randomcrop;
end

Augmentor(width::Int, height::Int, depth::Int) = Augmentor(RandomCrop(width, height, depth))

function (c::Augmentor)(img, mask)
    z = zeros(16, 1, 1)
    img = hcat(z, img, z)
    mask = hcat(z, mask, z)
    img, mask = augment(img, mask)
    img, mask = c.randomcrop(img, mask)
    img = reshape(img, (size(img)...,1))
    mask = reshape(mask, (size(mask)...,1))
    return img, mask
end

function flip(img, axis) 
    width, height, depth = size(img)
    if axis==1
        img = img[width:1:-1, :, :]
    elseif axis ==2
        img = img[:, height:1:-1, :]
    end
    return img
end

function rotate90(img)
    img = permutedims(img, (2, 1, 3))
    img = flip(img, 1)  # flip by 1 x 90
    return img
end

function rotate180(img)
    img = flip(img, 1)
    img = flip(img, 2)
    return img
end

function rotate270(img)
    img = permutedims(img, (2, 1, 3))
    img = flip(img, 2)  # flip by 3 x 90
    return img
end

function augment(img, mask)
    axis = rand(1:2)
    img, mask = flip.((img, mask), (axis, axis))
    angle = rand(0:3)
    if angle == 1
        img, mask = rotate90.((img, mask))
    elseif angle == 2
        img, mask = rotate180.((img, mask))
    elseif angle == 3 
        img, mask = rotate270.((img, mask))
    end
    # img, mask = 
    return img, mask
end