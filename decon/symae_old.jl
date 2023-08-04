
using Flux, MLUtils, Statistics, OneHotArrays, JuliennedArrays
xpu = gpu

"""
Returns data iterator `X` where each datapoint is generated by randomly picking  `ntau` instances from a ticular state.
During training, it is very important to mix instances from different states in each batch. In other words, a small batch size, e.g., `1`, is observed not to disentangle coherent from nuisance information. Therefore, it is recommended to set the batch size as high as possible.

* `nd` : number of output datapoints
* `ntau` : number of instances in each datapoint (typically 20)
* `dvec` : a vector where the elements are measured instances from different states
* `batchsize` : number of data points used in each batch for training
"""
function get_data_iterator(dvec; nd=1000, batchsize=256, ntau=20)
    drepeat = Flux.stack([randobs(randobs(dvec), ntau) for i = 1:nd], dims=3)
    return BatchView(drepeat, batchsize=batchsize)
end

## get networks
function get_dense_networks(nt, p, q; nt_senc=nt, nt_nenc=nt, nt_dec=nt)
    l2p = floor.(Int, LinRange(nt_senc, p, 5))
    lq = floor.(Int, LinRange(nt, q, 5))
    lpq = floor.(Int, LinRange(p + q, nt_dec, 5))
    senc =
        Chain(
            Dense(nt_senc, l2p[2], elu),
            Dense(l2p[2], l2p[3], elu),
            Dense(l2p[3], l2p[4], elu),
            Dense(l2p[4], p),
        ) |> xpu

    wsenc =
        Chain(
            Dense(p, p),
            x->softmax(x, dims=ndims(x) - 1) # sum to 1 along instance dimension
        ) |> xpu

    # senc2 = Chain(Dense(2 * p, 2 * p, elu), Dense(2 * p, p)) |> xpu

    nenc =
        Chain(
            Dense(nt_nenc, lq[2], elu),
            Dense(lq[2], lq[3], elu),
            Dense(lq[3], lq[4], elu),
            Dense(lq[4], q, elu),
        ) |> xpu

    # produce logvar for the nuisance encoder
    nenc_μ = Chain(Dense(q, q)) |> xpu
    nenc_logvar = Chain(Dense(q, q)) |> xpu

    dec =
        Chain(
            Dense(p + q, lpq[2], elu),
            Dense(lpq[2], lpq[3], elu),
            Dense(lpq[3], lpq[4], elu),
            Dense(lpq[4], nt_dec),
        ) |> xpu

    dec_var = xpu(1.0)
    return (; senc, wsenc, nenc, nenc_μ, nenc_logvar, dec, dec_var)
end
##

## get networks
function get_dense_networks_new(nt, p, q1, q2)
    l2p = floor.(Int, LinRange(nt, 2 * p, 5))
    l2q = floor.(Int, LinRange(nt, q1, 5))
    lq = floor.(Int, LinRange(nt * q1, q2, 5))
    lpq = floor.(Int, LinRange(p + q2, nt, 5))
    senc1 =
        Chain(
            Dense(nt, l2p[2], elu),
            Dense(l2p[2], l2p[3], elu),
            Dense(l2p[3], l2p[4], elu),
            Dense(l2p[4], 2 * p),
        ) |> xpu
    senc2 = Chain(Dense(2 * p, 2 * p, elu), Dense(2 * p, p)) |> xpu

    nenc1 =
        Chain(
            Dense(nt, l2q[2], elu),
            Dense(l2q[2], l2q[3], elu),
            Dense(l2q[3], l2q[4], elu),
            Dense(l2q[4], q1),
        ) |> xpu

    nenc2 =
        Chain(
            Dense(nt * q1, lq[2], elu),
            Dense(lq[2], lq[3], elu),
            Dense(lq[3], lq[4], elu),
            Dense(lq[4], q2),
            Dropout(0.5, dims=(1, 2, 3)),
        ) |> xpu

    dec =
        Chain(
            Dense(p + q2, lpq[2], elu),
            Dense(lpq[2], lpq[3], elu),
            Dense(lpq[3], lpq[4], elu),
            Dense(lpq[4], nt),
        ) |> xpu
    return (; senc1, senc2, nenc1, nenc2, dec)
end

## get networks
function get_dense_networks_new2(nt, p, q1, q2)
    l2p = floor.(Int, LinRange(nt, 2 * p, 5))
    l2q = floor.(Int, LinRange(nt, q1, 5))
    lq = floor.(Int, LinRange(nt * q1, q2, 5))
    lpq = floor.(Int, LinRange(p + q2, nt, 5))
    senc1 =
        Chain(
            Dense(nt, l2p[2], elu),
            Dense(l2p[2], l2p[3], elu),
            Dense(l2p[3], l2p[4], elu),
            Dense(l2p[4], 2 * p),
        ) |> xpu
    senc2 = Chain(Dense(2 * p, 2 * p, elu), Dense(2 * p, p)) |> xpu

    nenc1 = #[
        Chain(
            Dense(1, 2, elu),
            Dense(2, 2, elu),
            Dense(2, 2, elu),
            Dense(2, 2),
        )# for i in 1:nt] |> xpu

    nenc2 =
        Chain(
            Dense(nt * q1, lq[2], elu),
            Dense(lq[2], lq[3], elu),
            Dense(lq[3], lq[4], elu),
            Dense(lq[4], q2),
            Dropout(0.5, dims=(1, 2, 3)),
        ) |> xpu

    dec =
        Chain(
            Dense(p + q2, lpq[2], elu),
            Dense(lpq[2], lpq[3], elu),
            Dense(lpq[3], lpq[4], elu),
            Dense(lpq[4], nt),
        ) |> xpu
    return (; senc1, senc2, nenc1, nenc2, dec)
end
#
#
function get_conv_networks(nt, p, q)
    senc1 = Chain(
        x -> reshape(x, size(x, 1), 1, size(x, 2)),
        Conv((5,), 1 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        MaxPool((2,)),
    ) |> xpu

    senc2 = Chain(
        Conv((5,), 64 => 32, elu, ; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 32 => 1, ; pad=SamePad()),
        BatchNorm(1, elu),
        MaxPool((2,)),
        x -> Flux.flatten(x),
        Dense(div(nt, 16), p),
    ) |> xpu


    # 1D convolutional layers [time, channel, batch]
    nenc = Chain(
        x -> reshape(x, size(x, 1), 1, size(x, 2)),
        # Dropout(0.3, dims = (1, 2, 3)),
        Conv((5,), 1 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        # MaxPool((2,)),
        Dropout(0.4, dims=(1, 2, 3)),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 32, elu, ; pad=SamePad()),
        Dropout(0.4, dims=(1, 2, 3)),
        MaxPool((2,)),
        Conv((5,), 32 => 32, elu, ; pad=SamePad()),
        Conv((5,), 32 => 32, elu, ; pad=SamePad()),
        # MaxPool((2,)),
        Dropout(0.4, dims=(1, 2, 3)),
        Conv((5,), 32 => 8, elu, ; pad=SamePad()),
        Conv((5,), 8 => 1, ; pad=SamePad()),
        BatchNorm(1, elu),
        # MaxPool((2,)),
        x -> Flux.flatten(x),
        Dense(div(nt, 2), q),
        Dropout(0.4, dims=(1, 2, 3)),
    ) |> xpu



    dec = Chain(
        Dense(p + q, div(nt, 8), elu),
        x -> reshape(x, size(x, 1), 1, size(x, 2)),
        Conv((5,), 1 => 64, elu, ; pad=SamePad()),
        Upsample(8),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 64, ; pad=SamePad()),
        BatchNorm(64, elu),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 64, elu, ; pad=SamePad()),
        Conv((5,), 64 => 1, ; pad=SamePad()),
    ) |> xpu

    return (; senc1, senc2, nenc, dec)
end

struct BroadcastSenc
    senc1::Chain
    senc2::Chain
end
# Create batches of a time series
function batch_timeseries(X, s::Int, r::Int)
    T = size(X, 1)
    @assert s ≤ T "s cannot be longer than the total series"
    # X = X[((T - s) % r)+1:end]   # Ensure uniform sequence lengths
    # [selectdim(X, 1, t:r:T-s+t) for t ∈ 1:s] # Output
    [X[t:r:T-s+t, :, :] for t ∈ 1:s] # Output
end
function apply_shifts(x, onecoldarray)
    n1, n2, n3 = size(x)
    x1 = JuliennedArrays.Slices(x, True(), False(), False())
    oc1 = JuliennedArrays.Slices(onecoldarray, True(), False(), False())
    X = broadcast(x1, oc1) do xa, oc
        onecold(oc, batch_timeseries(xa, 50, 1))
    end
    reshape(stack(X, dims=(2)), (:, n2, n3))
end
struct BroadcastSenc1
    senc1::Chain
    senc2::Chain
end


struct BroadcastNenc
    chain::Chain
end

struct BroadcastTenc
    chain::Chain
end

struct BroadcastNenc_timeshift_invariant
    chain::Chain
end

struct BroadcastNenc_new{T}
    nenc1::Chain
    nenc2::Chain
    E::T
end
struct BroadcastNenc_new2{T}
    nenc1::T
    nenc2::Chain
end
struct BroadcastDec
    chain::Chain
end
struct JoinEncsDropout{T1,T2,T3}
    senc::T1
    nenc::T2
    nenc_drop::T3
end

function (m::BroadcastSenc)(x)
    x = cat(x, dims=3)
    n = size(x)
    X = reshape(x, n[1:end-2]..., n[end-1] * n[end])
    X = m.senc1(X)
    wX = m.senc2(X)

    X = X .* wX


    n1 = size(X)
    X = reshape(X, n1[1:end-1]..., n[end-1], n[end])
    X = mean(X, dims=ndims(X) - 1)
    X = dropdims(X, dims=ndims(X) - 1)
    # X = m.senc2(X)
    X = Flux.stack(fill(X, n[end-1]), dims=length(n) - 1)
    return X
end

function (m::BroadcastSenc1)(x)
    Xv = batch_timeseries(x, 100, 1)
    Xs = mapreduce((x, y) -> 0.5f0 .* (x + y), Xv) do x1
        x1 = cat(x1, dims=3)
        n = size(x1)
        X = reshape(x1, n[1:end-2]..., n[end-1] * n[end])
        X = m.senc1(X)
        n1 = size(X)
        X = reshape(X, n1[1:end-1]..., n[end-1], n[end])
        X = mean(X, dims=ndims(X) - 1)
        X = dropdims(X, dims=ndims(X) - 1)
        X = m.senc2(X)
        X = Flux.stack(fill(X, n[end-1]), dims=length(n) - 1)
        return X
    end
    return Xs
end
function (m::BroadcastNenc)(x)
    x = cat(x, dims=3)
    n1, n2, n3 = size(x)
    X = reshape(x, :, n2 * n3)
    X = m.chain(X)
    X = reshape(X, :, n2, n3)
    return X
end

function (m::BroadcastNenc_timeshift_invariant)(x)
    Xv = batch_timeseries(x, 50, 1)
    Xs = map(Xv) do x1
        x1 = cat(x1, dims=3)
        n1, n2, n3 = size(x1)
        X = reshape(x1, :, n2 * n3)
        X = m.chain(X)
        X = reshape(X, :, n2, n3)
        return X
    end
    return sum(Xs) / length(Xs)
end

function (m::BroadcastNenc_new)(x)
    x = cat(x, dims=3)

    X = mapreduce(vcat, m.E) do e
        m.nenc1(x .* e)
    end
    X = m.nenc2(X)
    return X
end

function (m::BroadcastNenc_new2)(x)
    x = cat(x, dims=3)
    @show size(x)
    n1, n2, n3 = size(x)
    x = chunk(x, n1, dims=1)
    X = mapreduce(vcat, x,) do xx
        xx = reshape(xx, 1, n2, n3)
        # @show size(xx)
        return m.nenc1(xx)
    end
    X = m.nenc2(X)
    return X
end
function (m::BroadcastDec)(x)
    x = cat(x, dims=3)
    n1, n2, n3 = size(x)
    X = reshape(x, :, n2 * n3)
    X = m.chain(X)
    X = reshape(X, :, n2, n3)
    return X
end
function (m::JoinEncsDropout)(x)
    x1 = m.senc(x)

    x = m.nenc(x)
    p = m.nenc_drop(x)

    # p = 0.5f0

    noise = xpu(randn(Float32, size(x)))

    x2 = x .+ (noise .* p)

    return cat(x1, x2, dims=1)
end
Flux.@functor BroadcastSenc
Flux.@functor BroadcastNenc
Flux.@functor BroadcastDec
Flux.@functor JoinEncsDropout





function partial_nenc(d)
    N = Zygote.jacobian(NN.nenc, d)
    n = NN.nenc(d)
    c = sencb(d)
    M = Zygote.jacobian(n) do nn
        NN.dec(cat(c, nn, dims=1))
    end
    return M[1] * N[1]
end


function partial_nenc(d0, d)
    N = Zygote.jacobian(NN.nenc, d)
    n = NN.nenc(d)
    c = sencb(d0)
    M = Zygote.jacobian(n) do nn
        NN.dec(cat(c, nn, dims=1))
    end
    return M[1] * N[1]
end

function partial_senc(d)
    N = Zygote.jacobian(sencb, d)
    n = NN.nenc(d)
    c = sencb(d)
    M = Zygote.jacobian(c) do cc
        NN.dec(cat(cc, n, dims=1))
    end
    return M[1] * N[1]
end

function norm_partial_senc(d)
    mapreduce(+, eachslice(d, dims=2)) do dd
        B = partial_senc(dd)
        return norm(B, 2) / sqrt(length(dd))
    end
end
function norm_partial_nenc(d)
    mapreduce(+, eachslice(d, dims=2)) do dd
        B = partial_nenc(dd)
        return norm(B, 2) / sqrt(length(dd))
    end
end