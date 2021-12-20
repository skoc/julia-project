include("modules.jl")

struct VNet;
    block_one;
    block_one_dw;
    block_two;
    block_two_dw;
    block_three;
    block_three_dw;
    block_four;
    block_four_dw;
    
    block_five;
    block_five_up;
    block_six;
    block_six_up;
    block_seven;
    block_seven_up;
    block_eight;
    block_eight_up;

    block_nine;
    out_conv; 
    out_conv2; 
end

function VNet(in_channels::Int=3, n_classes::Int=2, n_filters::Int=16, norm::Bool=false)
    if !norm
    println("No normalization is used.")
    VNet(
        RConvBlock(1, in_channels, n_filters),  # block_one
        DownsamplingRConv3d(n_filters, 2*n_filters, 2),

        RConvBlock(2, 2*n_filters, 2*n_filters),    # block_two
        DownsamplingRConv3d(2*n_filters, 4*n_filters, 2),

        RConvBlock(3, 4*n_filters, 4*n_filters),    # block_three
        DownsamplingRConv3d(4*n_filters, 8*n_filters, 2),

        RConvBlock(3, 8*n_filters, 8*n_filters),    # block_four
        DownsamplingRConv3d(8*n_filters, 16*n_filters, 2),

        RConvBlock(3, 16*n_filters, 16*n_filters),  # block_five
        UpsamplingRConvTranspose3d(16*n_filters, 8*n_filters, 2),

        RConvBlock(3, 8*n_filters, 8*n_filters),  # block_six
        UpsamplingRConvTranspose3d(8*n_filters, 4*n_filters, 2),

        RConvBlock(3, 4*n_filters, 4*n_filters),  # block_seven
        UpsamplingRConvTranspose3d(4*n_filters, 2*n_filters, 2),

        RConvBlock(2, 2*n_filters, 2*n_filters),  # block_eight
        UpsamplingRConvTranspose3d(2*n_filters, n_filters, 2),

        RConvBlock(1, n_filters, n_filters),
        Conv3d(n_filters, n_classes, 1, 0, 1)      
        Conv3d(n_filters, n_classes, 1, 0, 1)      
    )
    else
    println("Batch normalization is used.")
    VNet(
        BatchConvBlock(1, in_channels, n_filters),  # block_one
        DownsamplingBatchConv3d(n_filters, 2*n_filters, 2),

        BatchConvBlock(2, 2*n_filters, 2*n_filters),    # block_two
        DownsamplingBatchConv3d(2*n_filters, 4*n_filters, 2),

        BatchConvBlock(3, 4*n_filters, 4*n_filters),    # block_three
        DownsamplingBatchConv3d(4*n_filters, 8*n_filters, 2),

        BatchConvBlock(3, 8*n_filters, 8*n_filters),    # block_four
        DownsamplingBatchConv3d(8*n_filters, 16*n_filters, 2),

        BatchConvBlock(3, 16*n_filters, 16*n_filters),  # block_five
        UpsamplingBatchConvTranspose3d(16*n_filters, 8*n_filters, 2),

        BatchConvBlock(3, 8*n_filters, 8*n_filters),  # block_six
        UpsamplingBatchConvTranspose3d(8*n_filters, 4*n_filters, 2),

        BatchConvBlock(3, 4*n_filters, 4*n_filters),  # block_seven
        UpsamplingBatchConvTranspose3d(4*n_filters, 2*n_filters, 2),

        BatchConvBlock(2, 2*n_filters, 2*n_filters),  # block_eight
        UpsamplingBatchConvTranspose3d(2*n_filters, n_filters, 2),

        BatchConvBlock(1, n_filters, n_filters),
        Conv3d(n_filters, n_classes, 1, 0, 1)      
        Conv3d(n_filters, n_classes, 1, 0, 1)      
    )
    end

end

function (c::VNet)(x)
    # encode
    x1 = c.block_one(x)
    xx = c.block_one_dw(x1)

    x2 = c.block_two(xx)
    xx = c.block_two_dw(x2)

    x3 = c.block_three(xx)
    xx = c.block_three_dw(x3)

    x4 = c.block_four(xx)
    xx = c.block_four_dw(x4)

    x5 = c.block_five(xx)
    x5 = dropout(x5, 0.5)

    # decode
    xx = c.block_five_up(x5)
    xx = xx .+ x4

    x6 = c.block_six(xx)
    xx = c.block_six_up(x6)
    xx = xx .+ x3

    x7 = c.block_seven(xx)
    xx = c.block_seven_up(x7)
    xx = xx .+ x2

    x8 = c.block_eight(xx)
    xx = c.block_eight_up(x8)
    xx = xx .+ x1

    x9 = c.block_nine(xx)
    x9 = dropout(x9, 0.5)

    # output layer
    out = c.out_conv(x9)
    out_tanh = tanh.(x9)
    out_seg = c.out_conv2(x9)
    return out_tanh, out_seg    
end

# model = VNet(1, 3, 2)
# x = ones(32, 32, 32, 1, 11);
# y = model(x);
# size(y)