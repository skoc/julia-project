using Knet: logsoftmax, softmax

function kl_div(input_logits, target_logits)
    logsoft_input = logsoftmax(input_logits; dims=1)
    soft_target = softmax(target_logits; dims=1)
    logsoft_target = logsoftmax(target_logits; dims=1)
    return mean(soft_target .* (logsoft_target .- logsoft_input))
end

function mse_loss(y_hat, y)
    return mean((y - y_hat)^2)
end

function dice_loss(score, target)
    smooth = 1e-5
    intersect = sum(score .* target)
    y_sum = sum(target .* target)
    z_sum = sum(score .* score)
    loss = (2 .* intersect .+ smooth) / (z_sum + y_sum .+ smooth)
    loss = 1 .- loss
    return loss
end