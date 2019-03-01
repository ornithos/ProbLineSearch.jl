function probLineSearch(func::AbstractPLSFunction, search::PLSSearchDefn, pars::PLSConstantParams)
    x0 = evaluate(func, 0, search)
    history = PLSHistory(eltype(x0), pars.niter)
    history.recording = true
    push!(history.evals, 1)
    probLineSearch(func, x0, search, pars, history=history)
end


function probLineSearch(func::AbstractPLSFunction, x0::PLSEvaluation{T}, search::PLSSearchDefn{T};
        pars::PLSConstantParams{T}=PLSConstantParams(T), history::PLSHistory{T}=PLSHistory(T)) where T <: AbstractFloat

    @unpack α₀, search_direction, denom = search
    @unpack niter, c1, c2, wolfeThresh, verbosity = pars

    # Need to learn normalisation factor from x0, then store for rest of proc.
    norm∇_0    = norm(x0.∇y)
    if !isapprox(norm∇_0, 1.0)
        search.denom = norm∇_0   # magnitude for rescaling: |y′(0)|
        normalise_for_linesearch!(x0, search)
    else
        !isapprox(search.denom, 1.0) && (@warn "Perhaps x0 has been pre-normalised; and norm factor in searchpars≈1.0. This may cause problems.")
    end

    gp = PLSPosterior(zeros(T, 1), zeros(T, 1), [x0.∇y], x0.σ²_f, x0.σ²_∇) # T, Y, ∇Y, ΣY, Σ∇

    iterates = [x0]

    tt = T(1.0)   # initial step size (rescaled)

    for i = 1:niter
        #======================================================================
                            Evaluate chosen point.
         =====================================================================#
        x = evaluate(func, tt, search; history=history)
        push!(iterates, x)

        gp = update(gp, x)
        # -- check last evaluated point for acceptance ------------------------
        if probWolfe(gp, tt, c1, c2) > wolfeThresh   # are we done?
            display("bark1")
            push!(history.msg, "found acceptable point. (1).")
            finalise(x, gp, search, pars, history);
            return x, gp # done  => x.x, x.y, x.J, x.∇_J, search.α₀, history.ewma_α
        end
        #======================================================================
            Has updating the GP led to an existing point qualifying?
         =====================================================================#
        # ----------------- find minimum of posterior mean --------------------
        minM, minT, minj  = min_mean(gp)   # of existing points
        min∇ = d1m(gp, gp.Ts[minj])

        # -------------- ∇ ≈ 0 with high probability? -------------------------
        if abs(min∇) < 1e-5 && Vd(gp, minT) < 1e-4 # nearly deterministic
            display("bark2")
            push!(history.msg, "found a point with almost zero gradient. Stopping, although Wolfe conditions not guaranteed.")
            x = iterates[minj]
            finalise(x, gp, search, pars, history)
            return x
        end

        # -------------- Wolfe Conditions satisfied? --------------------------
        Tsort  = sort(gp.Ts);
        # ignore first point
        wolfes_ix = 1 .+ findall(probWolfe.(gp, Tsort[2:end], c1, c2) .> wolfeThresh)


        if !isempty(wolfes_ix)
            display("bark3")
            push!(history.msg, "found acceptable point. (2).")
            tt = Tsort[argmin(m(gp, Tsort[wolfes_ix]))]  # choose point with smallest post. mean.
            minj = findfirst(gp.Ts .== tt)
            x = iterates[minj]
            finalise(x, gp, search, pars, history)
            return x;
        end

        #======================================================================
            Choose next evaluation using EI/ProbWolfe via candidates
         =====================================================================#

        # -- CANDIDATES 1: minimal means between all evaluations.--------------

        # iterate through all `cells' (O[N]), check minimum mean locations.
        Tcand = zeros(T, 0); # positions of candidate points

        for cell = 1:length(gp)-1 # loop over cells
            _cellT_ϵ   = Tsort[cell] + 1e-6 * (Tsort[cell+1] - Tsort[cell]);
            cell_minT  = cubicMinimum(gp, _cellT_ϵ);

            # add point to candidate list if minimum lies in between T(cel) and T(cel+1)
            if Tsort[cell] < cell_minT < Tsort[cell+1];
                push!(Tcand, cell_minT);

            # ~~PATHOLOGY~~
            #= If unable to find minimum in first cell *AND* the estimated ∇ > 0
               it is possible that (assuming the search direction is ok) the scale
               is wrong, so return a point v. close to original, and reduce α₀.=#
           elseif cell==1 && d1m(gp, 0) > 0
               display("bark4")
                push!(history.msg, "function seems very steep, reevaluating close to start.")
                tt = 0.01 * (Tsort[cell] + Tsort[cell+1]);
                x = evaluate(func, tt, search, history=history)
                finalise(x, gp, search, pars, history)
                search.α₀ /= 10
                search.ewma_α /= 5
                return x  # done
            end
        end

        # -- CANDIDATES 2: one extrapolation step -----------------------------
        push!(Tcand, maximum(gp.Ts) + search.extrap_amt);
        display(Tcand)
        # display("before")
        # -- DISCRIMINATION: EI and ProbWolfe ---------------------------------
        # Calculate some statistics for EI.
        μs = m(gp, Tcand)         # means of candidate points
        σs = sqrt.(V(gp, Tcand))   # s.d. of candidate points
        # display(μs)
        # display(σs)
        # display(minM)
        # display(minT)
        # display(minj)
        EIcand = EI.(μs, σs, minM); # minM = η: best (lowest) GP mean of existing points

        PPcand = probWolfe.(gp, Tcand, c1, c2)
        idx_best = argmax(EIcand .* PPcand)

        # If best point is found using extrapolation, extend extrapolation for next iter.
        (idx_best == length(Tcand)) && (search.extrap_amt *= 2)
        display(EIcand)
        display(PPcand)
        display("Chosen value: $idx_best")
        # Next point chosen for evaluation (next iter!):
        tt = Tcand[idx_best];

        # makePlot();
    end

    #======================================================================
        !!! Not found an acceptable candidate in the niter limit.
     =====================================================================#
    x = evaluate(func, tt, search, history=history)
    push!(iterates, x)
    gp = update(gp, x)

    # -- check last evaluated point for acceptance ------------------------
    if probWolfe(gp, tt, c1, c2) > wolfeThresh   # are we done?
        display("bark5")
        push!(history.msg, "found acceptable point. (3).")
        finalise(x, gp, search, pars, history);
        return x # done  => x.x, x.y, x.J, x.∇_J, search.α₀, history.ewma_α
    end

    # -- return point with lowest mean ----------------------------------------
    # Code of Mahsereci and Hennig exclude t=0 from this, but I'm not sure I buy this.
    minM, minT, minj  = min_mean(gp)
    display("bark6")
    (verbosity > 0) && @warn "reached evaluation limit. Returning 'best' known point."
    push!(history.msg, "reached evaluation limit. Returning 'best' known point.")
    x = iterates[minj]
    finalise(x, gp, search, pars, history);
    return x

    # makePlot();
end






function finalise(x::PLSEvaluation{T}, post::PLSPosterior{T}, searchpars::PLSSearchDefn{T},
        pars::PLSConstantParams, history::PLSHistory{T}) where T <: AbstractFloat
    @unpack x₀ ,α₀, f₀, denom = searchpars
    @unpack αGrowth, verbosity, c1, c2 = pars

    # chosen step size
    α = x.t*α₀

    # Transform back to original frame of reference / undo [0,1] normalisation.
    invert_normalise_for_linesearch!(x, searchpars)

    # set new step size
    # next initial step size is αGrowth (def=1.3x) larger than last accepted step size
    push!(history.αs, α)
    searchpars.α₀ = α * αGrowth

    # running average for reset in case the step size becomes very small
    # this is a safeguard
    gamma = 0.95;
    history.ewma_α = gamma*history.ewma_α + (1-gamma)*α;

    # reset for next iter: any extrapolation should be baked into α₀
    searchpars.extrap_amt = 1

    # reset NEXT initial step size to average step size if accepted step
    # size is 100 times smaller or larger than average step size
    if (searchpars.α₀ > 1e2 * history.ewma_α)||(searchpars.α₀ < 1e-2 * history.ewma_α)
        (verbosity > 0) && println("v large/small value of α, resetting alpha0")
        searchpars.α₀ = history.ewma_α # reset step size
    end

    # Calculate acceptance statistics
    minM, _minT, _minj  = min_mean(post)   # of existing points
    push!(history.EI, EI(m(post, x.t), sqrt(V(post, x.t)), minM))
    push!(history.Wolfe, probWolfe(post, x.t, c1, c2))
    history.recording = false
end
