
@inline normalise_for_linesearch(f, ∇f, f₀, m, α₀) = (ft - f₀)/(α₀*m),  ∇f/m
@inline zT(N...) = zeros(T, N...)


function probLineSearch(func::AbstractPLSFunction, search::PLSSearchDefn, pars::PLSConstantParams)
    x0 = evaluate(func, 0, search)
    history = PLSTermination(eltype(x0), pars.niter))
    history.evals[1] = 1
    probLineSearch(func, x0, search, pars, history=history)
end


function probLineSearch(func::AbstractPLSFunction, x0::PLSEvaluation, search::PLSSearchDefn, pars::PLSConstantParams;
        history::PLSTermination=PLSTermination(eltype(x0), pars.niter))

    @unpack α₀, search_direction = search
    @unpack niter, c1, c2, wolfeThresh, verbosity = pars

    m    = norm(x0.∇y)

    # #####  => I'm not really sure that these are used. ######
        σ_f  = sqrt(x0.σ²_f)/(α₀ * m)
        σ_∇  = sqrt(σ²_∇)/m
    # #####  ........................................... ######

    search.m = m   # magnitude for rescaling: |y′(0)|
    gp = PLSPosterior(zT(1), zT(1), [x0.∇y/m], x0.σ²_f, x0.σ²_∇) # T, Y, ∇Y, ΣY, Σ∇
    pls_stats = PLSTermination(T, niter)

    iterates = [x0]

    tt = 1.0   # initial step size (rescaled)

    for i = 1:niter
        #======================================================================
                            Evaluate chosen point.
         =====================================================================#
        x = evaluate(func, tt, search)
        push!(iterates, x)

        gp = update!(gp, tt, x.y, x.∇y)

        # -- check last evaluated point for acceptance ------------------------
        if probWolfe(tt) > wolfeThresh   # are we done?
            history.reason = "found acceptable point. (1)."
            finalise(x, gp, search, pars, history);
            return x # done  => x.x, x.y, x.J, x.∇_J, search.α₀, history.ewma_α
        end

        #======================================================================
            Has updating the GP led to an existing point qualifying?
         =====================================================================#
        # ----------------- find minimum of posterior mean --------------------
        minM, minT, minj  = min_mean(gp)   # of existing points
        min∇ = d1m(gp, gp.Ts[minj])

        # -------------- ∇ ≈ 0 with high probability? -------------------------
        if abs(min∇) < 1e-5 && Vd(gp, minT) < 1e-4 # nearly deterministic
            history.reason = "found a point with almost zero gradient. Stopping, although Wolfe conditions not guaranteed."
            x = iterates[minj]
            finalise(x, gp, search, pars, history)
            return x
        end

        # -------------- Wolfe Conditions satisfied? --------------------------
        Tsort  = sort(gp.Ts);
        # ignore first point
        wolfes_ix = 1 .+ findall(probWolfe.(Tsort[2:end]) .> wolfeThresh)

        if !isempty(wolfes_ix)
            history.reason = "found acceptable point. (2)."
            tt = Tsort[argmin(m.(gp, Tsort[wolfes_ix]))]  # choose point with smallest post. mean.
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
        Tcand = []; # positions of candidate points

        for cell = 1:N-1 # loop over cells
            _cellT_ϵ   = Tsort[cell] + 1e-6 * (Tsort[cell+1] - Tsort[cell]);
            cell_minT  = cubicMinimum(_cellT_ϵ);

            # add point to candidate list if minimum lies in between T(cel) and T(cel+1)
            if Tsort[cell] < cell_minT < Tsort[cell+1];
                push!(Tcand, cell_minT);

            # ~~PATHOLOGY~~
            #= If unable to find minimum in first cell *AND* the estimated ∇ > 0
               it is possible that (assuming the search direction is ok) the scale
               is wrong, so return a point v. close to original, and reduce α₀.=#
            elseif cell==1 && d1m(gp, 0) > 0
                history.reason = "function seems very steep, reevaluating close to start."
                tt = 0.01 * (Tsort[cell] + Tsort[cell+1]);
                x = evaluate(func, tt, search)
                finalise(x, gp, search, pars, history)
                search.α₀ /= 20
                search.ewma_α /= 10
                return x  # done
            end
        end

        # -- CANDIDATES 2: one extrapolation step -----------------------------
        push!(Tcand, max(gp.Ts) + search.extrap_amt);

        # -- DISCRIMINATION: EI and ProbWolfe ---------------------------------
        # Calculate some statistics for EI.
        μs = m.(gp, Tcand)         # means of candidate points
        σs = sqrt.(V.(gp, Tcand)   # s.d. of candidate points
        EIcand = EI.(μs, σs, minM); # minM = η: best (lowest) GP mean of existing points

        PPcand = probWolfe.(gp, Tcand)
        idx_best = argmax(EIcand .* PPcand)

        # If best point is found using extrapolation, extend extrapolation for next iter.
        (idx_best == length(Tcand)) && (search.extrap_amt *= 2)

        # Next point chosen for evaluation (next iter!):
        tt = Tcand(idx_best);

        # makePlot();
    end

    #======================================================================
        !!! Not found an acceptable candidate in the niter limit.
     =====================================================================#
    x = evaluate(func, tt, search)
    push!(iterates, x)
    gp = update!(gp, tt, x.y, x.∇y)

    # -- check last evaluated point for acceptance ------------------------
    if probWolfe(tt) > wolfeThresh   # are we done?
        history.reason = "found acceptable point. (3)."
        finalise(x, gp, search, pars, history);
        return x # done  => x.x, x.y, x.J, x.∇_J, search.α₀, history.ewma_α
    end

    # -- return point with lowest mean ----------------------------------------
    # Code of Mahsereci and Hennig exclude t=0 from this, but I'm not sure I buy this.
    minM, minT, minj  = min_mean(gp)
    (verbosity > 0) && @warn "reached evaluation limit. Returning 'best' known point."
    history.reason = "reached evaluation limit. Returning 'best' known point."
    x = iterates[minj]
    finalise(x, gp, search, pars, history);
    return x

    # makePlot();
end






function finalise(x::PLSEvaluation{T}, post::PLSPosterior{T}, searchpars::PLSSearchDefn{T},
        pars::PLSConstantParams, history::PLSTermination{T}) where T <: AbstractFloat
    # !!!!!! THIS IS PROBABLY WRONG => WORK ALWAYS ASSUMES ACCEPTED PT WILL BE THE LAST EVALUATED !!!!!
    error("Assumes accepted pt is last evaluation. Not so sure?")
    @unpack x₀ ,α₀, f₀, m = searchpars
    @unpack αGrowth, verbosity = pars

    x.y = x.y*(α₀*m) + f₀              # function value at accepted position (undo normalisation tform)

    # set new step size
    # next initial step size is αGrowth (def=1.3x) larger than last accepted step size
    searchpars.α₀ = x.t*α₀ * αGrowth

    # running average for reset in case the step size becomes very small
    # this is a safeguard
    gamma = 0.95;
    history.ewma_α = gamma*history.ewma_α + (1-gamma)*tt*α₀;
    searchpars.extrap_amt = 1  # reset for next iter: any extrapolation should be baked into α₀

    # reset NEXT initial step size to average step size if accepted step
    # size is 100 times smaller or larger than average step size
    if (searchpars.α₀ > 1e2 * history.ewma_α)||(searchpars.α₀ < 1e-2 * history.ewma_α)
        (verbosity > 0) && println('v large/small value of α, resetting alpha0')
        searchpars.α₀ = history.ewma_α # reset step size
    end

    # Calculate acceptance statistics
    minM, _minT, _minj  = min_mean(post)   # of existing points
    history.EI = EI(m(post, x.t), sqrt(V(post, x.t)), minM)
    history.Wolfe = probWolfe(post, x.t)
end
