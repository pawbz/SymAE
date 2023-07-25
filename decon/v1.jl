### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7c9f2512-12aa-11ee-22c6-7f9abc5509a0
using Flux, JLD2, Random, MLUtils, DSP, ProgressLogging, Statistics, Zygote, LinearAlgebra, PlutoLinks, PlutoUI, PlutoHooks, OneHotArrays, HDF5, Tullio, Distributions, CUDAKernels, KernelAbstractions, CUDA, PlutoPlotly, FFTW, ImageFiltering, JuliennedArrays, StatsBase, LinearMaps, IterativeSolvers

# ╔═╡ 9e8a8120-a2d0-4898-b5f4-bd69b722fd39
TableOfContents()

# ╔═╡ 874fecca-4aad-46f2-b182-f3558e83fe5d
padfft(x, n) = cat(x, zeros(eltype(x), n - size(x, 1), size(x, 2)), dims=1)

# ╔═╡ b05a2c6c-b9c7-413b-8aa7-320f2cff9f00
function _unwrap(phase)
    samples = size(phase, 1)
    unwrapped = unwrap(phase, dims=(1))
    center = div(samples + 1, 2)
    ndelay = round.(unwrapped[center:center, :] ./ pi)
    unwrapped = unwrapped .- pi .* ndelay .* range(1, samples) ./ center
    return unwrapped
end

# ╔═╡ 88c44bce-0e8c-4873-83b8-385472a5e7cd
# function to remove linear phase
# need to do some testing
function remove_linear_phase(x)
    nfft = 20 * size(x, 1)
    nthalf = div(size(x, 1), 2)
    X = padfft(x, nfft)
    X = fft(X, 1)
    Xa = abs.(X)
    Xph = angle.(X)
    Xph = _unwrap(Xph)
    X = Xa .* cis.(Xph)
    xout = real.(ifft(X, 1))
    return vcat(xout[end-nthalf+1:end, :], xout[1:nthalf+1, :],)

end

# ╔═╡ 03b389ec-38df-4b85-8463-4d9fea910c5b
plot(cpu(D[1][:, 100]))

# ╔═╡ 9732fb0d-19de-4da8-b9d9-d1996d7bea92
md"## Earthquake Data"

# ╔═╡ 5e7917a8-31c2-42bb-9319-9658c76eb8ec
eqdata = h5open("eq_sample_data_deep.hdf5", "r");

# ╔═╡ f2de68d3-c15c-4041-8fd0-7ec5ac0a4fad
function taper(x)
    # x=circshift(x, (-50, 0))
    x = cat(tukey(size(x, 1), 0.1), dims=ndims(x)) .* x
    # n1, n2 = size(x)
    # cat(zeros_like(x, Float32, (50,n2)), x[51:end-50, :], zeros_like(x, Float32, (50,n2)), dims=1)
    return x
end

# ╔═╡ 1318a949-95bb-47a0-ab10-95483eea9925
Deqnames_syn = ["hnd", "kam", "pb2"]

# ╔═╡ 7e242126-5fbd-4835-9329-6f72de1c9e5c
md"## Synthetic Data"

# ╔═╡ 442bc064-1b10-406b-bab2-abbe683f6b3c
# begin
#     Daug_g = zeros(256, nsamp_train)
#     toy_green!(Daug_g, true)
# end;

# ╔═╡ 2ce80f50-cc99-4977-b1f7-68b9251d23ec
# Daug = augmented_state(data.D, x -> augmentfunc(x, 10, 120));

# ╔═╡ 156f4131-1e42-47f0-b785-023dc02cbd62
# Daug_win = gaussian_windows(size(data.D[1], 1), size(data.D[1], 2), σ);

# ╔═╡ f24202b9-aea3-4036-bd50-b06556a5b2ca
# Daug_win = augmented_state(size(data.D[1], 1), size(data.D[1], 2), x->augmented_state2d_spikes!(x,180, 220));
# Daug_win = augmented_state(size(data.D[1], 1), size(data.D[1], 2), x -> augmented_state2d_spikes!(x, 29, 30));

# ╔═╡ 988ec323-fc33-4eaa-b9e0-fdeb05ca5d8f
Daugloader = DataLoader(Daug[1], batchsize=2, shuffle=true);

# ╔═╡ 8854372d-f734-487f-a299-501fcd4bd21f
md"## Envelopes"

# ╔═╡ 6cb037f4-96ef-4067-a886-5faecf2e82a8
md"## Networks"

# ╔═╡ ebaee61b-9df2-4fee-b7cc-9c5ed25f8cdc
begin
    opt = ADAM()
    trainloss = []
    testloss = []
end;

# ╔═╡ 676642c4-8205-4523-b069-787be1040a5e
md"""
## Train
"""

# ╔═╡ 41f064cc-6c29-41cf-8123-577606d756ad
md"## Losses"

# ╔═╡ d267011b-7f39-4c1a-aeee-7e51b26cd06a
function update_losses(trainloss, testloss, NN)
    NN = map(x -> Flux.trainmode!(x, false), NN)
    NN = map(x -> Flux.testmode!(x, true), NN)
    dd1 = randobs(d1, 20)
    dd2 = randobs(d2, 20)
    l1 = Flux.mse(cat(model(dd1), model(dd2), dims=2), cat(dd1, dd2, dims=2))
    l2 =
        Flux.mse(cat(model(d1test), model(d2test), dims=2), cat(d1test, d2test, dims=2))
    push!(trainloss, l1)
    push!(testloss, l2)
    return l1, l2
end

# ╔═╡ 974459b1-18ba-4dcb-b403-ac1725364eec
md"## Redatuming"

# ╔═╡ ace79f71-abc9-4bd6-861f-859693ceca31
begin
    i = 2
    j = 4
end

# ╔═╡ 25fa4c7c-9ac7-4b8d-ba20-713ffbbd549e
md"## Plots"

# ╔═╡ b56b260d-5ef4-4571-a0f0-af169c543fc3
plot(randobs(xde))

# ╔═╡ b9550c88-1908-449b-8f5a-180e23cf1c8f
plot(mean(g1, dims=2))

# ╔═╡ c5d6576f-1b02-4228-ae09-1ff4c01a2e3e
plot(hcat(abs.(rfft(cpu(s1))), abs.(rfft(cpu(s2)))))

# ╔═╡ 9af909db-27ee-403a-be8d-0662926be389
function plot_data(d; title="", labels=fill(nothing, length(d)), ylims=(-8, 8), opacitys=fill(1, length(d)))
    trace = [PlutoPlotly.scatter(y=cpu(dd), mode="lines", name=label, opacity=opacity) for (dd, label, opacity) in zip(cpu(d), labels, opacitys)]
    layout = Layout(title=title, legend=attr(x=0, y=-0.3, traceorder="normal", orientation="h"), yaxis=attr(range=ylims, gridwidth=1), height=250)
    PlutoPlotly.plot(trace, layout)
end

# ╔═╡ 702b10d3-d727-41ee-8af9-45c17ef57bfe
begin
    s = map(1:50) do i
        if (i == 1)
            s1 = rpad(gaussian(10, 0.05, zerophase=false), 100, 0.0)
        elseif (i == 2)
            s1 = rpad([1.0], 100, 0.0)
        else

            s1 = rpad(randn(10), 100, 0.0)
        end
        return s1 #.- mean(s1)
    end
    # apply_gaussian3d!(S, t0=50, σ=0.1)
    # s1 = vec(imfilter(S[:, 1, :], Kernel.gaussian(2)))
    # s2 = vec(imfilter(S[:, 2, :], Kernel.gaussian(2)))
    s = map(s) do s1
        s1 = s1 * inv(std(s1))
    end

    plot_data([s[1], randobs(s)], title="Sources")
    ##
end

# ╔═╡ e44c9a2b-cc8e-4a2a-8be7-dfa3c184a9c0
plot((fftshift(s[4])))

# ╔═╡ 20793d1d-391e-46d9-8fef-19ee66fe2b59
let
    autos = mapreduce(hcat, s) do s1
        autocor(s1, 0:150)
    end
    plot(autos)
end

# ╔═╡ b131c111-f5a5-496d-9c93-4cc6d9b1c171
plot_data(unstack(randobs(randobs(Daug), 2), dims=2), title="Sample Augmented State")

# ╔═╡ e44a045e-1c10-4aba-a4f2-c931b3cd42f3
plot_data(unstack(randobs(Daug_win, 2), dims=2), title="Sample Augmented State")

# ╔═╡ 8339c5a0-6fad-451b-b4df-ed22588d806a
md"## Appendix"

# ╔═╡ 818c44ef-86bf-410d-b698-8cff6c6969b1
CUDA.device!(7)

# ╔═╡ d4034569-cd45-4a98-94d2-0de39491201e
symae = @ingredients("symae.jl")

# ╔═╡ 76b6baa5-b480-44c2-8249-35348bb8b412
nt1 = maximum(length.(symae.batch_timeseries(1:nt, 100, 1)))

# ╔═╡ f9740d59-a5be-4d96-83fd-5a23e30ce609
mapreduce((x, y) -> 0.5f0 .* (x + y), symae.batch_timeseries(1:nt, 5, 1)) do x
    x
end

# ╔═╡ 2f740254-f7e3-4cb3-8722-98489b3a5a33
conv = @ingredients("Conv.jl")

# ╔═╡ 061182ee-9068-411e-92dc-861c3210bc3b
xpu = symae.xpu

# ╔═╡ 9b9b2e5e-5da9-451d-95bf-f46de2d703ca
Deq_syn = map(["syn_data_hnd.hdf5", "syn_data_kam.hdf5", "syn_data_pb2.hdf5"]) do file
    data = Float32.(Array(h5open(file, "r")["data"]))
    Flux.normalise(xpu(taper(resample(data, 0.5, dims=1))), dims=1)[:, 1:1500]
end;

# ╔═╡ 928a1d48-bc3a-4fdc-99f6-f6e262456303
randobs(Deq_syn[3]) |> cpu |> plot

# ╔═╡ 151f75ce-facf-434a-9c77-83ea005cdac9
dataeq_syn = (; D=Deq_syn, nt=401, Dnames=Deqnames_syn, p=200, q=100);

# ╔═╡ f057ea85-82e7-443f-8d98-554cb1b7fbe7
begin
    Deq = map(keys(eqdata)) do eq
        mapreduce(hcat, keys(eqdata[eq])) do pixel
            resampled_data = resample(Array(eqdata[eq][pixel]["data"]), 0.5, dims=1)
            # resampled_zero_lag_data = remove_linear_phase(resampled_data)
            Flux.normalise(xpu(taper(resampled_data)), dims=1)
        end
    end
    Deqnames = keys(eqdata)
    Deq = vcat(Deq, Deq_syn)
    Deqnames = vcat(Deqnames, Deqnames_syn)
end

# ╔═╡ 7ae0922f-0575-4c91-91dc-53e71cafd3bf
randobs(Deq[3]) |> cpu |> plot

# ╔═╡ 1ccc0ec1-798b-42cb-a91c-5f6b06820ca7
dataeq = (; D=Deq, nt=401, Dnames=Deqnames, p=200, q=50);

# ╔═╡ c3dce92a-46dc-4af8-816b-c1880858b85b
envelope(x) = abs.(hilbert(x))
# function envelope(x) 
# x = abs.(rfft(x))
# return x ./ std(x)
# end

# ╔═╡ 029baae4-1ea8-4656-93b3-b97f9c2f261d
plot()

# ╔═╡ 35ed8f63-9b30-4d87-b6a6-96c1553a0689
md"### Toy Green's Function"

# ╔═╡ 34c33216-ef0f-4ec9-a98c-4042d12b33af
# 
function toy_green!(x, first_arrival, second_arrival)
    nt = size(x, 1)
    nr = size(x, 2)
	arrival_std = 0.5
	
    for ir = 1:nr
        # first arrival
        it0 = round(Int, rand(truncated(Normal(first_arrival, arrival_std), 1, nt)))
        x[it0, ir] = randn()
        
        # last arrival - randomly present between first two
        it1 = round(Int, rand(truncated(Normal(second_arrival, arrival_std), 1, nt)))
        x[it1, ir] = randn()
    end
    x = x .- mean(x, dims=1)
    return x
end

# ╔═╡ 77f1447c-377e-433e-8663-69a64d2de141
begin
    nsamp_train = 500
    nsamp_test = 100
    g = map(1:length(s)) do i
        g1 = zeros(128, nsamp_train)
        second_arrival = sample([60])
        toy_green!(g1, 35, second_arrival)

		# g1 = mapreduce(hcat, -10:10) do tshift
		# 	circshift(g1, (tshift, 0))
		# end
        return g1
    end
    # end
    #    g1train, g1test = splitobs(g1, at=0.9)
    #    g2 = zeros(300, nsamp_train)
    #    toy_green!(g2, true)
    #    g2train, g2test = splitobs(g2, at=0.9)
    # gnew = zeros(nt, nsamp_test)
    # toy_green!(gnew, true)
    plot_data(unstack(randobs(randobs(g), 5), dims=2), title="randomly picked filters", ylims=(-2, 2))
    ##
end

# ╔═╡ 4337f434-ecf9-4e14-b5f6-d9659c46ab09
function multiconv(g, s)
    g = cat(g, dims=3)
    g = permutedims(g, [1, 3, 2])
    s = cat(s, dims=3)
    # s=permutedims(s,[1,3,2]); 
    d = Flux.conv(g, s, pad=size(s, 1), flipped=false)
    d = permutedims(d, [1, 3, 2])
    d = dropdims(d, dims=3)
    return Float32.(d[2:size(g, 1)+1, :])
end

# ╔═╡ ee3642fe-1489-4ce4-8ba5-f156367878fb
begin
    ##
    Dsyn = map(s, g) do s1, g1
        d = multiconv(g1, s1) |> xpu
        # d = d .- mean(d, dims=1)
        d = d ./ std(d, dims=1)
    end

    Dsynnames = ["State $i" for i in 1:length(Dsyn)]
end;

# ╔═╡ 3b291a4e-2c18-4c24-bfc1-8a148da1b474
datasyn = (; D=Dsyn, nt=128, Dnames=Dsynnames, p=100, q=25);

# ╔═╡ 8a474346-9e81-42a0-ac21-ad908f6425fb
md"""## Select Data
$(@bind data Select([datasyn=>"Synthetic Toy", dataeq=>"Real Data", dataeq_syn=>"Synthetic Earthquakes"]))
"""

# ╔═╡ 371f8fea-2bf3-4245-92cc-8116cf965260
D1loader = DataLoader(data.D[1], batchsize=128, shuffle=true);

# ╔═╡ 4fc2dd8b-fa92-4de6-a94d-558d1d6c8273
D2loader = DataLoader(data.D[4], batchsize=128, shuffle=true);

# ╔═╡ 543b3e59-3583-49fd-9900-a55fb80129c3
let
    dplot = cpu(randobs(randobs(data.D), 2))
    plot_data(unstack(dplot, dims=2), title="Sample Data", opacitys=[1, 1, 0.5, 0.5])
end

# ╔═╡ ac9e99ea-c83d-4b9b-a17e-1d53db25f0cc
De = map(data.D) do D1
    e = mapreduce(+, eachslice(cpu(D1), dims=2)) do d
        abs.(hilbert(d))
    end
    e = e / maximum(e)
end

# ╔═╡ a32830cf-3a01-44eb-8d34-0035fc98240c
begin
    # NN = symae.get_dense_networks(nt, p, q, nt_senc=nt, nt_nenc=nt1, nt_dec=2*nt-nt1)
    NN = symae.get_dense_networks(data.nt, data.p, data.q, true)

    sencb = symae.BroadcastSenc(NN.senc, NN.fsenc, NN.wsenc)
    # nencb = symae.BroadcastNenc_timeshift_invariant(NN.nenc)
    nencb = symae.BroadcastNenc(NN.nenc, NN.nenc_μ, NN.nenc_logσ)
	# nencb_logvar = symae.BroadcastNenc(NN.nenc_logvar)
    decb = symae.BroadcastDec(NN.dec, NN.dec_logvar)
    reconstruct = symae.Reconstruct(sencb, nencb, decb)
    # model(x) = symae.apply_shifts(decb(encb(x)), tencb(x))

end;

# ╔═╡ e8585118-4c68-40f5-9e1c-2b3acb841af6
NN.senc

# ╔═╡ d5dc689b-ffc4-42c8-aecf-46499e86b268
NN.wsenc

# ╔═╡ 1f1fbfc8-cc18-4cf9-a72f-3d28825adcb0
NN.nenc

# ╔═╡ 9fd00cca-ab31-485d-b756-2c9090d828f5
NN.dec

# ╔═╡ ab139277-786b-43be-8289-617d208e9b04
function loss_mlvae(x)
	
	cx, nμ, nlogσ, xhat, xhat_logvar = reconstruct(x)
	
    neg_log_likelihood = 0.5f0 * sum(@. (abs2(xhat - x) / exp(xhat_logvar)) + xhat_logvar)
	
	kl_nui =  0.5f0 * sum(@. (exp(2f0 * nlogσ) + nμ^2 -1f0 - 2f0 * nlogσ))

	return neg_log_likelihood + kl_nui
end

# ╔═╡ d1ac5752-b77c-4d11-8a22-0c3d273e87ea
loss = loss_mlvae

# ╔═╡ b886179f-3fa1-496f-9f3f-fd9ec2ece093
function loss_mse(x)
	cx, nμ, nlogσ, xhat, xhat_logvar = reconstruct(x)
	return Flux.mse(xhat, x)
end

# ╔═╡ b4416bfe-4b5c-4d0c-9864-25a50f720646
# use virtual data where source is used from x and nuisance from xaug
function generate_virtual_data(x, xaug)
    c = selectdim(sencb(x), 2, 1)
    C = Flux.stack(fill(c, size(xaug, 2)), dims=2)
	nx, _ = nencb(xaug)
    xhat, _ = decb(cat(C, nx, dims=1))
	return xhat
end

# ╔═╡ c93d0fd0-28ff-462e-8e59-c94b52531bf2
function redatum(d1, d2)

    _, _, _, d1hat, _ = reconstruct(d1, false)
    _, _, _, d2hat, _ = reconstruct(d2, false)

    d12hat = generate_virtual_data(d1, d2)
    d21hat = generate_virtual_data(d2, d1)

    return map(cpu, (; d1=d1, d2=d2, d1hat, d2hat, d12hat, d21hat))
end

# ╔═╡ 16b09cdc-824f-4426-9f06-3e2da38181e0
exp.(decb.logvar)

# ╔═╡ 5d41fd67-07b9-49d4-bdeb-f2a459065159
plot_data(De, ylims=(0, 1), labels=data.Dnames)

# ╔═╡ 28b4d791-acd8-4f40-8c34-838af73fbac4
md"""
Redatuming peak $(@bind iplt2 Slider(1:size(data.D[1], 2), show_value=true));
Gaussian standard deviation
$(@bind σ Select([0.0005, 0.001,  0.005, 0.01, 0.1, 1])); Plot Envelope? $(@bind penvelope CheckBox())
"""

# ╔═╡ 5cb4b1b5-a97b-43e1-9f5f-7de3f8dee769
data.Dnames

# ╔═╡ ae8dbe65-b1d1-4923-94f5-0c559dab9934
let
    xaug = randobs(Daug[2], 1)
    # Random.seed!(2)
    x = randobs(data.D[2], 1)


    @show size(x), size(xaug)
    xhat = decb(cat(sencb(x), nencb(x), dims=1))
    xdeconv = redatum_n(x, fill(xaug, 1), 1)
    xdeconv_x2 = redatum_n(xdeconv, [xaug], 1)
    # @show sencb(xdeconv)
    plot_data([vec(x[:, 1, :]), vec(xaug[:, 1, :]), vec(xhat[:, 1, :]), vec(xdeconv[:, 1, :]), vec(xdeconv_x2[:, 1, :]), vec(s2), vec(sencb(xdeconv)), vec(sencb(randobs(data.D[2]))), vec(nencb(xdeconv)), vec(nencb(xaug))], labels=["x", "xaug", "xhat", "xdeconv", "xdeconv_x2", "s2", "cod1", "cod2", "cod3", "cod4"])
    # plot_data([])
    # decb(sencb()
end

# ╔═╡ 8c8f7790-6111-4531-8d3b-adc192757f77
md"### Augmentation"

# ╔═╡ 7096c697-0f09-4406-b216-22fa36c5a4d7
function augmented_state3d!(x)
    randn!(x)
    nt, n2, n3 = size(x)
    x = map(eachslice(x, dims=(2, 3))) do xx
        t0 = rand(1:size(x, 1))
        σ = rand(Uniform(0.01, 0.05))
        # @tullio x[i,j,k] *= exp(-0.5f0*(abs2((i-t0) * inv(σ)))) # Gaussian?
        @tullio xx[i] *= (abs(i - t0) / (σ * div(nt, 2)) < 1) ? 1.0f0 : 0.0f0 # Rect
        # compute std
        stdx = std(xx)
        # normalise using std
        @tullio xx[i] *= inv(stdx)
    end
    return reshape(stack(x, dims=(2)), (:, n2, n3))
end

# ╔═╡ 36734cde-5605-47de-96db-2fe2ed04b96a
function augmented_state2d_random!(x, itmin, itmax)
    randn!(x)
    nt, n2 = size(x)
    x = map(eachslice(x, dims=(2))) do xx
        t0 = rand(itmin:itmax)
        σ = rand(Uniform(0.01, 0.05))
        # @tullio x[i,j,k] *= exp(-0.5f0*(abs2((i-t0) * inv(σ)))) # Gaussian?
        @tullio xx[i] *= (abs(i - t0) / (σ * div(nt, 2)) < 1) ? 1.0f0 : 0.0f0 # Rect
        # compute std
        stdx = std(xx)
        # normalise using std
        @tullio xx[i] *= inv(stdx)
    end
    return reshape(stack(x, dims=(2)), (:, n2))
end

# ╔═╡ 01e865d0-272b-439a-9154-81b289648b7d
function augmented_state2d_gaussian!(x, itmin, itmax)
    nt, n2 = size(x)
    x = map(eachslice(x, dims=(2))) do xx
        E = abs.(hilbert(cpu(xx)))
        t0 = sample(itmin:itmax, aweights(E)[itmin:itmax])
        σ = rand(Uniform(0.001, 0.01)) * div(nt, 2)
        @tullio xx[i] *= exp(-0.5f0 * (abs2((i - t0) * inv(σ)))) # Gaussian
        # compute std
        stdx = std(xx)
        # normalise using std
        @tullio xx[i] *= inv(stdx)
    end
    return reshape(stack(x, dims=(2)), (:, n2))
end

# ╔═╡ 41099e08-8d15-4efb-8af1-8301d9ff459f
function augmented_state2d_spikes!(x, itmin, itmax)
    nt, n2 = size(x)
    x = map(eachslice(x, dims=(2))) do xx
        t0 = sample(itmin:itmax)#, aweights(E)[itmin:itmax])
		fill!(xx, zero(eltype(x)))
        CUDA.@allowscalar xx[t0] = randn()
    
        # compute std
        stdx = std(xx)
        # normalise using std
        @tullio xx[i] *= inv(stdx)
		return xx
    end
    return reshape(stack(x, dims=(2)), (:, n2))
end

# ╔═╡ fa791701-3c50-4f5d-b144-da9bdae6d89e
md"""## Augmented State
Activate $(@bind augment CheckBox(default=false)) Select type? $(@bind augmentfunc Select([augmented_state2d_spikes! =>"Spikes", augmented_state2d_gaussian! =>"Gaussian", augmented_state2d_random! =>"Random"]))

"""

# ╔═╡ 235ec0aa-5ef6-47ce-85b2-b96407266cd3
if (augment)
    X = symae.get_data_iterator(vcat(data.D, Daug), batchsize=64, ntau=50)
else
    X = symae.get_data_iterator(data.D, batchsize=64, ntau=50)
end

# ╔═╡ 0046f186-a278-453b-9513-34a9a3ebacaa
loss(first(X))

# ╔═╡ 845daeed-b4f7-4675-9d63-144668d5d7a3
function update(nepoch, NN)
    # p = Progress(nepoch, showspeed = true)
    ps = Flux.params(values(NN)...)
    @progress name = "training" for epoch = 1:nepoch
        for x in X
            # x = apply_random_time_shifts!(x, -50:50)
            # gs = Flux.gradient(() -> loss(x), ps) # compute gradient
            gs = Flux.gradient(() -> loss(x), ps)

            # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters

        end
    end
    return NN
end

# ╔═╡ 2fd9a503-cf69-4c80-8af1-ac68a7a18886
trained = @use_memo([]) do
    update(100, NN)
    true
end

# ╔═╡ 0e899d57-0eb4-4d20-816c-3055ae186295
begin
    trained
    redatumed = redatum(data.D[i], data.D[j])
    # redatumed = redatum(Daug[i], data.D[j])
    # redatumed = redatum(data.D[i], Daug_win)
end

# ╔═╡ fa7fca08-3051-4931-bcf1-7002013c4a08

let
    mse3 = mapreduce(hcat, 1:2) do ip
        loss_nuisance_enforcer(data.D[1], data.D[2], xpu(selectdim(redatumed.d2newhat, 2, ip)))
    end
    plot(log10.(mse3'))
end

# ╔═╡ 23ff86a1-106f-4c34-95fb-8a371962a1d2
let
    mse1 = map(1:2) do ip
        loss_coherent_enforcer(data.D[1], xpu(selectdim(redatumed.d1newhat, 2, ip)))
    end
    plot(log10.(mse1)
    )
end


# ╔═╡ 12acb01e-29f0-4c64-8834-2c766cdf400a
let
    mse2 = map(1:2) do ip
        loss_coherent_enforcer(data.D[2], xpu(selectdim(redatumed.d2newhat, 2, ip)))
    end
    plot(log10.(mse2))
end

# ╔═╡ c15504cb-3dc4-465e-b7f4-fc1f188e2438
plot(hcat(abs.(rfft(cpu(redatumed.d1newhat)[:, iplt2])), abs.(rfft(cpu(redatumed.d2newhat)[:, iplt2]))))

# ╔═╡ 58ad1103-1ede-40d8-91db-45a17f311552
plot(real.(ifft(fft(cpu(redatumed.d1newhat)[:, iplt2]) ./ (1.0f-10 .+ fft(g2[:, iplt2])))))

# ╔═╡ 72e70ad4-b955-459c-9377-9802ba9bf6fb
begin
    trained
    plot_data([redatumed.d1[:, iplt2], redatumed.d1hat[:, iplt2]], title="$(data.Dnames[i]) reconstruction"; labels=["true", "reconstructed"])
end

# ╔═╡ a21cf4fd-3e86-4ff9-9446-4825bfaedfac
begin
    trained
    plot_data([redatumed.d2[:, iplt2], redatumed.d2hat[:, iplt2]], title="$(data.Dnames[j]) reconstruction", labels=["true", "reconstructed"])
end

# ╔═╡ 56ca859e-fb7f-4ff9-a77d-f98635b626f4
let
    trained
    d1newhat = redatumed.d12hat[:, iplt2]
    dvec = map([s[i], d1newhat, redatumed.d2[:, iplt2]]) do d
        penvelope ? envelope(d) : d
    end
    ylims = penvelope ? (0, 10) : (-8, 8)
    plot_data(dvec, title="Coherent: $(data.Dnames[i]); Nuisance: $(data.Dnames[j])", labels=["true", "virtual", "redatuming peak"], ylims=ylims, opacitys=[0.5, 1, 0.5])
end

# ╔═╡ 114880f0-825a-48ec-92df-69a84a2a9492
let
    trained
    d2newhat = redatumed.d21hat[:, iplt2]
    dvec = map([s[j], d2newhat, redatumed.d1[:, iplt2]]) do d
        penvelope ? envelope(d) : d
    end
    ylims = penvelope ? (0, 10) : (-8, 8)
    plot_data(dvec, title="Coherent: $(data.Dnames[j]); Nuisance: $(data.Dnames[i])", labels=["true", "virtual", "redatuming peak"], ylims=ylims, opacitys=[0.5, 1, 0.5])
end

# ╔═╡ ca752d09-37f9-4b1d-9d63-c6e3e1d78fff
let
    trained
    p = mapreduce(hcat, 4:4) do k
        x = data.D[7]
        # x = Daug_win
        # x = Daug[1]
        x1 = data.D[k]
        # Random.seed!(2)
        xde = generate_virtual_data(x1, x)

        virtual_data = cpu(xde[:, :, 1])
        impulsive_source_data = cpu(x)

		pa_conv=conv.Conv.Pconv(Float32, dsize=size(virtual_data), gsize=size(impulsive_source_data), ssize=(data.nt,), g=impulsive_source_data, d=virtual_data)
		paA=conv.Conv.operator(pa_conv, conv.Conv.G())
		s=zeros(Float32, data.nt); IterativeSolvers.lsmr!(s, paA, vec(virtual_data))
		return penvelope ? envelope(s) : s
        # xcor = fftshift(lucy(X1, X2, iterations=21))
        # return cpu(randobs(xde[:, :, 1]))
        # return randobs(X1)
        # return mean(abs.(hilbert(X1)), dims=2)
        # xcor = fftshift(real.(ifft((fft(X1, 1)) ./ (fft(X2, 1)), 1)), 1)
        # xcorp = mean(cpu(xcor), dims=2)
        # xcorp = 
        # return xcorp
    end
    plot(p)
end

# ╔═╡ 1f28b6ae-bb9d-4cf2-9f45-3a9e9b4a9be3
exp.(reconstruct(first(X), false)[3])

# ╔═╡ 0401e0d5-f14d-4a40-92d4-90e322901b11
function augmented_state(D, func)
    D1 = map(deepcopy(D)) do d
        func(d)
        return d
    end
end

# ╔═╡ 91577acb-6a82-4266-a7c5-172409a6adf6
function augmented_state(nt, nr, func)
    d = xpu(zeros(nt, nr))
    func(d)
end

# ╔═╡ 49ed3c9b-1ea1-4615-ad1f-8196289ccf71
function gaussian_windows(nt, nr, σ)
    if (nr < nt)
        iss = 1-div(nt, 2):nr-div(nt, 2)
    else
        iss = 1-div(nt, 2):nt-div(nt, 2)
        iss1 = [randobs(-div(nt, 2):div(nt, 2)) for i in 1:nr-nt]
        iss = vcat(iss, sort(iss1))
    end
    iss = -10:10
    dnew = mapreduce(hcat, iss) do is
        # circshift(gaussian(nt, σ), is + div(1.5*nt,4))
        circshift(gaussian(nt, σ), is)
    end
    dnew = Flux.normalise(dnew, dims=1)
    return xpu(dnew)
end

# ╔═╡ 4d3c8d2f-3feb-41b1-9180-7b358a914b6b
function my_gaussian(nt, σ)
    x = zeros(Float32, nt, 1, 1)
    augmented_state3d!(x)
    return vec(x)
end

# ╔═╡ d88d3f6d-6ba9-4589-b4f6-6015c5f470f8
# function apply_random_time_shifts(data, shifts)
# n1, n2, n3 = size(data)
# data1=map(eachslice(data, dims=(2, 3))) do x
# 	circshift(x, rand(shifts))
# end
# return reshape(stack(data1, dims=(2)), (:, n2, n3))
# end

# ╔═╡ 0f99968e-d1da-41ec-b27e-4f56415362e8
function apply_random_time_shifts!(data, shifts)
    n1, n2 = size(data)
    data1 = map(eachslice(data, dims=(2))) do x
        shift = rand(shifts)
        if (shift == 0)
            y = x
        elseif (shift > 0)
            y = cat(zeros_like(x, Float32, (shift)), x[1:end-shift], dims=1)
        else
            shift = abs(shift)
            y = cat(x[1+shift:end], zeros_like(x, Float32, (shift)), dims=1)
        end
        return copyto!(x, y)
    end
    return reshape(stack(data1, dims=(2)), (:, n2))

end

# ╔═╡ 9f410c60-dae6-42bd-8fd0-828d2c6bd82a
function append_shifted_state(D, shifts)
    D1 = map(D) do d
        dshifted = deepcopy(d)
        apply_random_time_shifts!(dshifted, shifts)
        return cat(d, dshifted, dims=2)
    end
end

# ╔═╡ 3d1d1462-47c5-4685-8196-86a9aeb49b21
begin
    fshift = x -> append_shifted_state(x, -10:10)
    Deqshifted = fshift(fshift(fshift(Deq)))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDAKernels = "72cfdca4-0801-4ab0-bf6a-d52aa10adc57"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
JuliennedArrays = "5cadff95-7770-533d-a838-a1bf817ee6e0"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Tullio = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CUDA = "~4.0.1"
CUDAKernels = "~0.4.7"
DSP = "~0.7.8"
Distributions = "~0.25.96"
FFTW = "~1.7.1"
Flux = "~0.13.17"
HDF5 = "~0.16.15"
ImageFiltering = "~0.7.5"
IterativeSolvers = "~0.9.2"
JLD2 = "~0.4.31"
JuliennedArrays = "~0.4.0"
KernelAbstractions = "~0.8.6"
LinearMaps = "~3.10.1"
MLUtils = "~0.4.3"
OneHotArrays = "~0.2.4"
PlutoHooks = "~0.0.5"
PlutoLinks = "~0.1.6"
PlutoPlotly = "~0.3.7"
PlutoUI = "~0.7.51"
ProgressLogging = "~0.1.4"
StatsBase = "~0.33.21"
Tullio = "~0.3.5"
Zygote = "~0.6.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "4a1e27a68be335ebef3719b3ef0c428c692f2fbb"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "edff14c60784c8f7191a62a23b15a421185bc8a8"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.0.1"

[[deps.CUDAKernels]]
deps = ["Adapt", "CUDA", "KernelAbstractions", "StaticArrays", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "1680366a69e9c95744ef23a239e6cfe61cf2e1ca"
uuid = "72cfdca4-0801-4ab0-bf6a-d52aa10adc57"
version = "0.4.7"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "75d7896d1ec079ef10d3aee8f3668c11354c03a1"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.2.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "d6b227a1cfa63ae89cb969157c6789e36b7c9624"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.1.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "ed00f777d2454c45f5f49634ed0a589da07ee0b0"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.2.4+1"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2918fbffb50e3b7a0b9127617587afa76d4276e8"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.8.1+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d730914ef30a06732bdd9f763f6cc32e92ffbff1"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "da8b06f89fce9996443010ef92572b193f8dca1f"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.8"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "4ed4a6df2548a72f66e03f3a285cd1f3b573035d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.96"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e17cc4dc2d0b0b568e80d937de8ed8341822de67"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote", "cuDNN"]
git-tree-sha1 = "3e2c3704c2173ab4b1935362384ca878b53d4c34"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.17"

    [deps.Flux.extensions]
    AMDGPUExt = "AMDGPU"
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "478f8c3145bb91d82c2cf20433e8c1b30df454cc"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "a3351bc577a6b49297248aadc23a4add1097c2ac"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.7.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "19d693666a304e8c371798f4900f7435558c7cde"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.3"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "c73fdc3d9da7700691848b78c61841274076932a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.15"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "3b20c3ce9c14aedd0adca2bc8c882927844bd53d"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.0+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SnoopPrecompile", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "d90867cbe037730a73c9a9499b3591eedbe387a0"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.JuliennedArrays]]
git-tree-sha1 = "d290d10d4aca892e2e09d3219658f3c562dab152"
uuid = "5cadff95-7770-533d-a838-a1bf817ee6e0"
version = "0.4.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "cf9cae1c4c1ff83f6c02cfaf01698f05448e8325"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.8.6"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f044a2796a9e18e0531b9b3072b0019a61f264bc"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.17.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "070e4b5b65827f82c16ae0916376cb47377aa1b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.18+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "a1348b9b7c87d45fa859314d56e8a87ace20561e"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.10.1"
weakdeps = ["ChainRulesCore"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8a5b4d2220377d1ece13f49438d71ad20cf1ba83"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.1.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "d86a788b336e8ae96429c0c42740ccd60ac0dfcc"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.8"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "6979eccb6a9edbbb62681e158443e79ecc0d056a"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "a8027af3d1743b3bfae34e54872359fdebb31422"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.3+4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "33ad5a19dc6730d592d8ce91c14354d758e53b0e"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.19"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "f3080f4212a8ba2ceb10a34b938601b862094314"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cae3153c7f6cf3f069a853883fd1919a6e5bab5b"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.9+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6a01f65dd8583dee82eecc2a19b0ff21521aa749"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.18"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5a6ab2f64388fd1175effdf73fe5933ef1e0bac0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Colors", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "90b12392675690592f9d1a29af1689d6c345f97e"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "dbde6766fc677423598138a5951269432b0fcc90"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.7"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.Tullio]]
deps = ["ChainRulesCore", "DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "7871a39eac745697ee512a87eeff06a048a7905b"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.5"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ead6292c02aab389cb29fe64cc9375765ab1e219"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "ec954b59f6b0324543f2e3ed8118309ac60cb75b"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.0.3"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eddd19a8dea6b139ea97bdc8a0e2667d4b661720"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.0.6+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═9e8a8120-a2d0-4898-b5f4-bd69b722fd39
# ╠═88c44bce-0e8c-4873-83b8-385472a5e7cd
# ╠═874fecca-4aad-46f2-b182-f3558e83fe5d
# ╠═b05a2c6c-b9c7-413b-8aa7-320f2cff9f00
# ╠═03b389ec-38df-4b85-8463-4d9fea910c5b
# ╠═76b6baa5-b480-44c2-8249-35348bb8b412
# ╠═f9740d59-a5be-4d96-83fd-5a23e30ce609
# ╟─9732fb0d-19de-4da8-b9d9-d1996d7bea92
# ╠═5e7917a8-31c2-42bb-9319-9658c76eb8ec
# ╠═f2de68d3-c15c-4041-8fd0-7ec5ac0a4fad
# ╠═9b9b2e5e-5da9-451d-95bf-f46de2d703ca
# ╠═1318a949-95bb-47a0-ab10-95483eea9925
# ╠═f057ea85-82e7-443f-8d98-554cb1b7fbe7
# ╠═928a1d48-bc3a-4fdc-99f6-f6e262456303
# ╠═7ae0922f-0575-4c91-91dc-53e71cafd3bf
# ╠═3d1d1462-47c5-4685-8196-86a9aeb49b21
# ╟─7e242126-5fbd-4835-9329-6f72de1c9e5c
# ╠═702b10d3-d727-41ee-8af9-45c17ef57bfe
# ╠═77f1447c-377e-433e-8663-69a64d2de141
# ╠═ee3642fe-1489-4ce4-8ba5-f156367878fb
# ╟─8a474346-9e81-42a0-ac21-ad908f6425fb
# ╠═1ccc0ec1-798b-42cb-a91c-5f6b06820ca7
# ╠═151f75ce-facf-434a-9c77-83ea005cdac9
# ╠═3b291a4e-2c18-4c24-bfc1-8a148da1b474
# ╟─fa791701-3c50-4f5d-b144-da9bdae6d89e
# ╠═442bc064-1b10-406b-bab2-abbe683f6b3c
# ╠═2ce80f50-cc99-4977-b1f7-68b9251d23ec
# ╠═156f4131-1e42-47f0-b785-023dc02cbd62
# ╠═f24202b9-aea3-4036-bd50-b06556a5b2ca
# ╠═235ec0aa-5ef6-47ce-85b2-b96407266cd3
# ╠═371f8fea-2bf3-4245-92cc-8116cf965260
# ╠═4fc2dd8b-fa92-4de6-a94d-558d1d6c8273
# ╠═988ec323-fc33-4eaa-b9e0-fdeb05ca5d8f
# ╠═543b3e59-3583-49fd-9900-a55fb80129c3
# ╠═b131c111-f5a5-496d-9c93-4cc6d9b1c171
# ╠═e44a045e-1c10-4aba-a4f2-c931b3cd42f3
# ╟─8854372d-f734-487f-a299-501fcd4bd21f
# ╠═ac9e99ea-c83d-4b9b-a17e-1d53db25f0cc
# ╟─6cb037f4-96ef-4067-a886-5faecf2e82a8
# ╠═ebaee61b-9df2-4fee-b7cc-9c5ed25f8cdc
# ╠═a32830cf-3a01-44eb-8d34-0035fc98240c
# ╠═0046f186-a278-453b-9513-34a9a3ebacaa
# ╠═e8585118-4c68-40f5-9e1c-2b3acb841af6
# ╠═d5dc689b-ffc4-42c8-aecf-46499e86b268
# ╠═1f1fbfc8-cc18-4cf9-a72f-3d28825adcb0
# ╠═9fd00cca-ab31-485d-b756-2c9090d828f5
# ╟─676642c4-8205-4523-b069-787be1040a5e
# ╠═2fd9a503-cf69-4c80-8af1-ac68a7a18886
# ╠═d1ac5752-b77c-4d11-8a22-0c3d273e87ea
# ╠═845daeed-b4f7-4675-9d63-144668d5d7a3
# ╟─41f064cc-6c29-41cf-8123-577606d756ad
# ╠═d267011b-7f39-4c1a-aeee-7e51b26cd06a
# ╠═ab139277-786b-43be-8289-617d208e9b04
# ╠═b886179f-3fa1-496f-9f3f-fd9ec2ece093
# ╟─974459b1-18ba-4dcb-b403-ac1725364eec
# ╠═b4416bfe-4b5c-4d0c-9864-25a50f720646
# ╠═c93d0fd0-28ff-462e-8e59-c94b52531bf2
# ╠═ace79f71-abc9-4bd6-861f-859693ceca31
# ╠═0e899d57-0eb4-4d20-816c-3055ae186295
# ╠═1f28b6ae-bb9d-4cf2-9f45-3a9e9b4a9be3
# ╠═16b09cdc-824f-4426-9f06-3e2da38181e0
# ╟─25fa4c7c-9ac7-4b8d-ba20-713ffbbd549e
# ╟─5d41fd67-07b9-49d4-bdeb-f2a459065159
# ╟─72e70ad4-b955-459c-9377-9802ba9bf6fb
# ╟─a21cf4fd-3e86-4ff9-9446-4825bfaedfac
# ╟─28b4d791-acd8-4f40-8c34-838af73fbac4
# ╟─56ca859e-fb7f-4ff9-a77d-f98635b626f4
# ╟─114880f0-825a-48ec-92df-69a84a2a9492
# ╠═5cb4b1b5-a97b-43e1-9f5f-7de3f8dee769
# ╠═ca752d09-37f9-4b1d-9d63-c6e3e1d78fff
# ╠═e44c9a2b-cc8e-4a2a-8be7-dfa3c184a9c0
# ╠═20793d1d-391e-46d9-8fef-19ee66fe2b59
# ╠═fa7fca08-3051-4931-bcf1-7002013c4a08
# ╠═23ff86a1-106f-4c34-95fb-8a371962a1d2
# ╠═12acb01e-29f0-4c64-8834-2c766cdf400a
# ╠═b56b260d-5ef4-4571-a0f0-af169c543fc3
# ╠═c15504cb-3dc4-465e-b7f4-fc1f188e2438
# ╠═b9550c88-1908-449b-8f5a-180e23cf1c8f
# ╠═c5d6576f-1b02-4228-ae09-1ff4c01a2e3e
# ╠═58ad1103-1ede-40d8-91db-45a17f311552
# ╠═9af909db-27ee-403a-be8d-0662926be389
# ╠═ae8dbe65-b1d1-4923-94f5-0c559dab9934
# ╟─8339c5a0-6fad-451b-b4df-ed22588d806a
# ╠═7c9f2512-12aa-11ee-22c6-7f9abc5509a0
# ╠═818c44ef-86bf-410d-b698-8cff6c6969b1
# ╠═d4034569-cd45-4a98-94d2-0de39491201e
# ╠═2f740254-f7e3-4cb3-8722-98489b3a5a33
# ╠═061182ee-9068-411e-92dc-861c3210bc3b
# ╠═c3dce92a-46dc-4af8-816b-c1880858b85b
# ╠═029baae4-1ea8-4656-93b3-b97f9c2f261d
# ╟─35ed8f63-9b30-4d87-b6a6-96c1553a0689
# ╠═34c33216-ef0f-4ec9-a98c-4042d12b33af
# ╠═4337f434-ecf9-4e14-b5f6-d9659c46ab09
# ╟─8c8f7790-6111-4531-8d3b-adc192757f77
# ╠═7096c697-0f09-4406-b216-22fa36c5a4d7
# ╠═36734cde-5605-47de-96db-2fe2ed04b96a
# ╠═01e865d0-272b-439a-9154-81b289648b7d
# ╠═41099e08-8d15-4efb-8af1-8301d9ff459f
# ╠═0401e0d5-f14d-4a40-92d4-90e322901b11
# ╠═91577acb-6a82-4266-a7c5-172409a6adf6
# ╠═9f410c60-dae6-42bd-8fd0-828d2c6bd82a
# ╠═49ed3c9b-1ea1-4615-ad1f-8196289ccf71
# ╠═4d3c8d2f-3feb-41b1-9180-7b358a914b6b
# ╠═d88d3f6d-6ba9-4589-b4f6-6015c5f470f8
# ╠═0f99968e-d1da-41ec-b27e-4f56415362e8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
