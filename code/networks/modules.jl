using Knet: Knet, conv4, pool, mat, nll, accuracy, progress, sgd, param, param0, dropout, relu, bnmoments, bnparams
using Knet: bnmoments, bnparams, deconv4,tanh

#=
Conv3d
=#
struct Conv3d;
    w;
    b;
    padding;
    stride;
end

(c::Conv3d)(x) = conv4(c.w, x; padding=c.padding, stride=c.stride, mode=1) .+ c.b # size(x) = (a, b, c, out_channels, batch)
Conv3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    Conv3d(param(kernel_size, kernel_size, kernel_size, in_channels, out_channels), 
    param0(1,1,1,out_channels,1), padding, stride)

#=
BatchNorm3d
=#
struct BatchNorm3d; 
    moments;
    params;
end

#=
Conv3d + batchnorm3d + relu
=#
(c::BatchNorm3d)(x) = batchnorm(x, c.moments, c.params) # size(x) = (a, b, c, out_channels, batch)
BatchNorm3d(out_channels::Int) = BatchNorm3d(bnmoments(), bnparams(out_channels))

struct BatchConv3d;
    conv3d;
    batch3d;
end

(c::BatchConv3d)(x) = relu.(c.batch3d(c.conv3d(x)))
BatchConv3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    BatchConv3d(Conv3d(in_channels, out_channels, kernel_size, padding, stride), BatchNorm3d(out_channels))

#=
Conv3d + relu
=#
struct RConv3d;
    conv3d;
end

(c::RConv3d)(x) = relu.(c.conv3d(x))
RConv3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    RConv3d(Conv3d(in_channels, out_channels, kernel_size, padding, stride))



#=
Sequential
=#
struct Sequential;
    layers;
end

function (c::Sequential)(x)
    for layer in c.layers
        x = layer(x)
    end
    x
end

#=
Sequence of Conv3d + relu
=#
struct RConvBlock;
    sequential;
end

function (c::RConvBlock)(x)
    x = c.sequential(x)
end

function RConvBlock(n_stages::Int, n_filters_in::Int, n_filters_out::Int)
    seq = [RConv3d(n_filters_in, n_filters_out, 3, 1, 1)]
    for i=2:n_stages
        push!(seq, RConv3d(n_filters_out, n_filters_out, 3, 1, 1))
    end
    return RConvBlock(Sequential(seq))
end

#=
Sequence of Conv3d + batchnorm3d + relu
=#
struct BatchConvBlock;
    sequential;
end

function (c::BatchConvBlock)(x)
    x = c.sequential(x)
end

function BatchConvBlock(n_stages::Int, n_filters_in::Int, n_filters_out::Int)
    seq = [BatchConv3d(n_filters_in, n_filters_out, 3, 1, 1)]
    for i=2:n_stages
        push!(seq, BatchConv3d(n_filters_out, n_filters_out, 3, 1, 1))
    end
    return BatchConvBlock(Sequential(seq))
end

#=
Residual Sequence of Conv3d + batchnorm3d + relu
=#
struct ResidualBatchConvBlock;
    sequential;
    conv3d;
    batchnorm3d;
end

function (c::ResidualBatchConvBlock)(x)
    y = c.sequential(x)
    y = c.conv3d(y)
    y = c.batchnorm3d(y)
    x = x .+ y
    x = relu.(x)
    return x
end

function ResidualBatchConvBlock(n_stages::Int, n_filters_in::Int, n_filters_out::Int)
    seq = [BatchConv3d(n_filters_in, n_filters_out, 3, 1, 1)]
    for i=2:n_stages-1
        push!(seq, BatchConv3d(n_filters_out, n_filters_out, 3, 1, 1))
    end
    return ResidualBatchConvBlock(Sequential(seq), Conv3d(n_filters_out, n_filters_out, 3, 1, 1), BatchNorm3d(n_filters_out))
end

#=
Residual Sequence of Conv3d + relu
=#
struct ResidualRConvBlock;
    sequential;
    conv3d;
end

function (c::ResidualRConvBlock)(x)
    y = c.sequential(x)
    y = c.conv3d(y)
    x = x .+ y
    x = relu.(x)
    return x
end

function ResidualRConvBlock(n_stages::Int, n_filters_in::Int, n_filters_out::Int)
    seq = [RConv3d(n_filters_in, n_filters_out, 3, 1, 1)]
    for i=2:n_stages-1
        push!(seq, RConv3d(n_filters_out, n_filters_out, 3, 1, 1))
    end
    return ResidualRConvBlock(Sequential(seq), Conv3d(n_filters_out, n_filters_out, 3, 1, 1))
end

#=
Downsampling layer: Conv3d + relu
=#
struct DownsamplingRConv3d;
    conv3d;
end

(c::DownsamplingRConv3d)(x) = relu.(c.conv3d(x))
DownsamplingRConv3d(in_channels::Int, out_channels::Int, stride::Int) = 
    DownsamplingRConv3d(Conv3d(in_channels, out_channels, stride, 0, stride))

#=
Downsampling layer: Conv3d + batchnorm3d + relu
=#
struct DownsamplingBatchConv3d;
    conv3d;
    batch3d;
end

(c::DownsamplingBatchConv3d)(x) = relu.(c.batch3d(c.conv3d(x)))
DownsamplingBatchConv3d(in_channels::Int, out_channels::Int, stride::Int) = 
    DownsamplingBatchConv3d(Conv3d(in_channels, out_channels, stride, 0, stride), BatchNorm3d(out_channels))

#=
Transposed convolution
=#
struct ConvTranspose3d;
    w;
    b;
    padding;
    stride;
end

(c::ConvTranspose3d)(x) = deconv4(c.w, x; padding=c.padding, stride=c.stride, mode=1) .+ c.b # size(x) = (a, b, c, out_channels, batch)
ConvTranspose3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    ConvTranspose3d(param(kernel_size, kernel_size, kernel_size, out_channels, in_channels), param0(1,1,1,out_channels,1), padding, stride)


#=
Sequence of ConvTranspose3d + relu
=#
struct RConvTranspose3d;
    convtranspose3d;
end

(c::RConvTranspose3d)(x) = relu.(c.convtranspose3d(x))
RConvTranspose3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    RConvTranspose3d(ConvTranspose3d(in_channels, out_channels, kernel_size, padding, stride))

#=
Sequence of ConvTranspose3d + batchnorm3d + relu
=#
struct BatchConvTranspose3d;
    convtranspose3d;
    batch3d;
end

(c::BatchConvTranspose3d)(x) = relu.(c.batch3d(c.convtranspose3d(x)))
BatchConvTranspose3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    BatchConvTranspose3d(ConvTranspose3d(in_channels, out_channels, kernel_size, padding, stride), BatchNorm3d(out_channels))


#=

=#
struct RConvTranspose3d;
    convtranspose3d;
end

(c::RConvTranspose3d)(x) = relu.(c.convtranspose3d(x))
RConvTranspose3d(in_channels::Int, out_channels::Int, kernel_size::Int, padding::Int, stride::Int) = 
    RConvTranspose3d(ConvTranspose3d(in_channels, out_channels, kernel_size, padding, stride))

#=

=#
struct UpsamplingRConvTranspose3d;
    convtranspose3d;
end

(c::UpsamplingRConvTranspose3d)(x) = relu.(c.convtranspose3d(x))
UpsamplingRConvTranspose3d(in_channels::Int, out_channels::Int, stride::Int) = 
    UpsamplingRConvTranspose3d(ConvTranspose3d(in_channels, out_channels, stride, 0, stride))