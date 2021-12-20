using Random;

function sample(labeled_indices, unlabeled_indices, labeled_bs, unlabeled_bs)
    labeled_indices = shuffle(labeled_indices)
    i = 1
    batches = []
    while i<=length(labeled_indices)
        lb = i+labeled_bs-1
        if lb > length(labeled_indices)
            lb = length(labeled_indices)
        end
        push!(batches, (labeled_indices[i:lb], shuffle(unlabeled_indices)[1:unlabeled_bs]))
        i = i+labeled_bs
    end
    batches
end