struct SGD
    lr;
    momentum;
    weight_decay;
    v;
end

function (c::SGD)(model, loss)
    i = 1
    for p in params(model)
        c.v[i] = c.v[i] .* c.momentum + (1-c.momentum) .* (grad(loss, p) + 2 * c.weight_decay .* p)
        p = p .- c.lr * c.v[i]        
        i = i + 1
    end
end

function SGD(lr::Int, momentum::Float, weight_decay::Float, model)
    v = []
    for p in params(model)
        push!(v, zeros(size(p)...))
    end
    SGD(lr, momentum, weight_decay, v)
end