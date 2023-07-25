

function loss_correlation(d1, daug)
    dv = generate_virtual_data(daug, d1)
    C = map(Iterators.product(unstack(dv, dims=2), unstack(d1, dims=2))) do (ddv, dd1)
        dot(ddv, dd1)
    end
    Cbar = diagm(ones_like(diag(C)))
    return Flux.mse(C, Cbar)
end


function loss_nuisance_enforcer2(d1, d2, daug)

    dv1 = generate_virtual_data(daug, d1)
    dvv1 = generate_virtual_data(d1, dv1)
    dv2 = generate_virtual_data(daug, d2)
    dvv2 = generate_virtual_data(d2, dv2)

    return Flux.mse(dvv1, d1) + Flux.mse(dvv2, d2)
end

function loss_coherent_enforcer(d1, daug)
    daug = DDaug
    dv = generate_virtual_data(d1, daug)
    dvv = generate_virtual_data(dv, d1)
    Flux.mse(dvv, d1)
end

function loss_nuisance_enforcer(d1, d2, daug, reduce_flag=false)
    daug = DDaug
    dv1 = generate_virtual_data(d1, daug)
    dvv1 = generate_virtual_data(daug, dv1)
    dv2 = generate_virtual_data(d2, daug)
    dvv2 = generate_virtual_data(daug, dv2)
    E = [Flux.mse(dvv1, daug), Flux.mse(dvv2, daug)]
    if (reduce_flag)
        return sum(E)
    else
        return E
    end



    # dv = generate_virtual_data(d1, daug)
    # dvv = generate_virtual_data(d1, dv)
    # Flux.mse(dvv, dv)

    # dv1 = generate_virtual_data(d1, daug)
    # dv11 = generate_virtual_data(d1, dv1)

    # dv2 = generate_virtual_data(d2, daug)
    # dv22 = generate_virtual_data(d2, dv2)

    # dv12 = generate_virtual_data(d1, dv2)
    # dv21 = generate_virtual_data(d2, dv1)
    # Flux.mse(dv12, dv1) + Flux.mse(dv21, dv2) + Flux.mse(dv11, dv1) + Flux.mse(dv2, dv22)

    # Flux.mse(dvv, dv)
end


function model2(x)

	S = redatum_mean_coherent(x)
	N = redatum_mean_nuisance(x)

    s1 = sencb(N)
    n1 = nencb(S)


    return decb(cat(s1, n1, dims=1))

end


function redatum_mean_coherent(x)
	s = sencb(x)
	n = nencb(x)
    ssize = size(s)
	s0 = mean(s, dims=2:3)
	s0 = zero(s0)
	s0 = repeat(s0, 1, ssize[2], ssize[3])
	return decb(cat(s0, n, dims=1))
end


function redatum_mean_nuisance(x)
	s = sencb(x)
    n = nencb(x)
    nsize = size(n)
    # n0 = fill(mean(n), nsize)
	n0 = mean(n, dims=2:3)
	n0 = zero(n0)
	n0 = repeat(n0, 1, nsize[2], nsize[3])
	
   	return decb(cat(s, n0, dims=1))
end

       #  for (d1, d2, daug) in zip(D1loader, D2loader, Daugloader)
        #          # x = apply_random_time_shifts!(x, -50:50)
        #          # gs = Flux.gradient(() -> loss(x), ps) # compute gradient
        # gs = Flux.gradient(() -> loss_coherent_enforcer(d1, daug) + loss_coherent_enforcer(d2, daug), ps)
        #          Flux.Optimise.update!(opt, ps, gs) # update parameters
        #      end
        # ========

        # 				 for (d1, d2, daug) in zip(D1loader, D2loader, Daugloader)
        #             # x = apply_random_time_shifts!(x, -50:50)
        #             # gs = Flux.gradient(() -> loss(x), ps) # compute gradient
        # 			gs = Flux.gradient(() -> loss_nuisance_enforcer2(d1, d2, daug), ps)
        #             Flux.Optimise.update!(opt, ps, gs) # update parameters
        #         end
        # # ========


        # ltrain, ltest = update_losses(trainloss, testloss, NN)
        # ProgressMeter.next!(
        #     p;
        #     showvalues = [(:epoch, epoch), (:train_loss, ltrain), (:test_loss, ltest)],
        # )


        function redatum_n(x, xaug, cycle_len)
            xout = copy(x)
            for i in 1:cycle_len
                xout = decb(cat(sencb(xout), nencb(xaug[i]), dims=1))
            end
            return xout
        end