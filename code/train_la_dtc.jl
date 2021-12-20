include("./networks/vnet.jl")
include("./dataloader/dataset.jl")
include("./dataloader/sampler.jl")
include("./dataloader/augmentation.jl")
include("./optimizer.jl")
include("./loss.jl")
include("./utils.jl")

using Knet: sigmoid, bce 


num_classes = 2
in_channels = 1
n_filters= 16
norm = true

batch_size = 4
labeled_bs = 2
unlabeled_bs = batch_size - labeled_bs
data_path = "/home/ayoub/Desktop/Projects/semi-supervised-julia/pytorch/DTC/data"
labelnum = 16
labeled_idxs = 1:labelnum
unlabeled_idxs = labelnum+1:80
patch_size = (112, 112, 80)

base_lr = 0.01
momentum = 0.9
weight_decay = 0.0001

max_epoch = 10

model = VNet(in_channels, num_classes-1, n_filters, norm)
images, labels = read_dataset(data_path, true)
augmentor = Augmentor(patch_size...)
sgd = SGD(base_lr, momentum, weight_decay, model)


for epoch in 1:max_epoch
    batch_indices = sample(labeled_idxs, unlabeled_idxs, labeled_bs, unlabeled_bs)
    for labeled, unlabeled in batch_indices
    # sample a batch
    x, y = images[labeled], labels[labeled] # TODO
    # augment batch
    x, y = augmentor.(x, y)
    x, y = cat(x, dims=5), cat(y, dims=5)

    # compute model output
    outputs_tanh, outputs = model(x)
    outputs_soft = sigmoid.(outputs)
    
    # compute ground-truth vallue
    gt_dis = compute_sdf(labels, size(outputs[:,:,:,1,1:labeled_bs]))

    # compute loss
    mloss = mse_loss(outputs_tanh[:,:,:,1, 1:labeled_bs], gt_dis)
    closs = bce(outputs[:,:,:,1,:labeled_bs], y[:,:,:,1,:labeled_bs])


    # optimize model

    # update learning rate if necessary

    # log metrics

end

    
end