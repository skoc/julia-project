function compute_sdf(masks, out_shape)
    normalized_sdf = zeros(out_shape)
    for batch in 1:size(out_shape)[5]
        posmask = masks[:,:,:,:, batch] .!= 0
        if any(posmask)
            
        end
    end
end