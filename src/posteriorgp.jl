
"""
Summary of posterior over (f, f′) given current evaluations. The posterior
takes the form of a GP, which can be queried using methods #TODO, and stores
the evaluations so far.
"""

const ζ = 10     # numerical stability in kernel function (any positive const is ok)

mutable struct PLSPosterior{T <: AbstractFloat}
    Ts::Vector{T}
    Y::Vector{T}
    ∇Y::Vector{T}
    σ²_f::T
    σ²_∇::T
    G::Matrix{T}   # posterior kernel matrix
    A::Vector{T}   # posterior mean weights
end

# Constructor
function PLSPosterior(Ts::Vector{T}, Y::Vector{T}, ∇Y::Vector{T}, σ²_f::T, σ²_∇::T) where T <: AbstractFloat

    N = length(Ts)

    # build prior Gram matrix components
    K   = k.(Ts, Ts')
    K∂  = kd.(Ts, Ts')
    ∂K∂ = dkd.(Ts, Ts')

    # build full (noised) Gram matrix
    G = [K K∂; transpose(K∂) ∂K∂];
    G[diagind(G)[1:N]] .+= σ²_f
    G[diagind(G)[N+1:2N]] .+= σ²_∇

    # weights of posterior mean (for linear combo of query points)
    A = G \ [Y; ∇Y];

    return PLSPosterior(Ts, Y, ∇Y, σ²_f, σ²_∇, G, A)
end


#= Integrated Wiener Process Kernel functions (function and derivative)
------------------------------------------------------------------------=#
# kernel:
k(a,b)   = min(a + ζ,b + ζ)^3/3 + abs(a-b) * min(a + ζ,b + ζ)^2/2;
kd(a,b)  = (a<b) * ((a + ζ)^2/2) + (a>=b) * ((a + ζ)*(b + ζ) - (b + ζ)^2/2);
dk(a,b)  = (a>b) * ((b+ζ)^2/2) + (a<=b) .* ((a+ζ)*(b+ζ) - (a+ζ).^2/2);
dkd(a,b) = min(a+ζ,b+ζ);

# further derivatives
ddk(a,b) = (a<=b) * (b-a);
ddkd(a,b) = (a<=b);
dddk(a,b) = -(a<=b);


#= Posterior statistic calculations for PLSPosterior
------------------------------------------------------------------------=#
# posterior mean function and all its derivatives
m(gp::PLSPosterior, t::Vector)   = let Ts = gp.Ts; [k.(t, Ts')    kd.(t,  Ts')  ] * gp.A; end
d1m(gp::PLSPosterior, t::Vector) = let Ts = gp.Ts; [dk.(t, Ts')   dkd.(t,  Ts') ] * gp.A; end
d2m(gp::PLSPosterior, t::Vector) = let Ts = gp.Ts; [ddk.(t, Ts')  ddkd.(t,  Ts')] * gp.A; end
d3m(gp::PLSPosterior{T}, t::Vector) where T <: AbstractFloat = let Ts = gp.Ts; [dddk.(t, Ts')  zeros(T, length(t), length(Ts))] * gp.A; end

# posterior marginal covariance between function and first derivative
Σ(gp::PLSPosterior, t::Vector)   = let Ts = gp.Ts; k.(t,t')   - ([k.(t, Ts')   kd.(t, Ts') ] * (gp.G \ [k.(t, Ts')   kd.(t, Ts') ]')); end
Σd(gp::PLSPosterior, t::Vector)  = let Ts = gp.Ts; kd.(t,t')  - ([k.(t, Ts')   kd.(t, Ts') ] * (gp.G \ [dk.(t, Ts')  dkd.(t, Ts')]')); end
dΣd(gp::PLSPosterior, t::Vector) = let Ts = gp.Ts; dkd.(t,t') - ([dk.(t, Ts')  dkd.(t, Ts')] * (gp.G \ [dk.(t, Ts')  dkd.(t, Ts')]')); end

V(gp::PLSPosterior, t::Vector) = diag(Σ(gp, t))
Vd(gp::PLSPosterior, t::Vector) = diag(Σd(gp, t))
dVd(gp::PLSPosterior, t::Vector) = diag(dΣd(gp, t))

# covariance terms with function (derivative) values at origin
# TODO: vectorised versions (d>1) do not work! (dimensions must match)
Σ0f(gp::PLSPosterior, t::Vector)   = let Ts = gp.Ts; k.(0,t)   - ([k.(0, Ts')   kd.(0, Ts') ] * (gp.G \ [k.(t, Ts')   kd.(t, Ts') ]')); end
Σd0f(gp::PLSPosterior, t::Vector)  = let Ts = gp.Ts; dk.(0,t)  - ([dk.(0, Ts')  dkd.(0, Ts')] * (gp.G \ [k.(t, Ts')   kd.(t, Ts') ]')); end
Σ0df(gp::PLSPosterior, t::Vector)  = let Ts = gp.Ts; kd.(0,t)  - ([k.(0, Ts')   kd.(0, Ts') ] * (gp.G \ [dk.(t, Ts')  dkd.(t, Ts')]')); end
Σd0df(gp::PLSPosterior, t::Vector) = let Ts = gp.Ts; dkd.(0,t) - ([dk.(0, Ts')  dkd.(0, Ts')] * (gp.G \ [dk.(t, Ts')  dkd.(t, Ts')]')); end

V0f(gp::PLSPosterior, t::Vector) = diag(Σ0f(gp, t))
Vd0f(gp::PLSPosterior, t::Vector) = diag(Σd0f(gp, t))
V0df(gp::PLSPosterior, t::Vector) = diag(Σ0df(gp, t))
Vd0df(gp::PLSPosterior, t::Vector) = diag(Σd0df(gp, t))

# # ========================= SCALAR VERSIONS =============================
m(gp::PLSPosterior, t::Real) = m(gp, [t]) |> arr2sc
d1m(gp::PLSPosterior, t::Real) = d1m(gp, [t]) |> arr2sc
d2m(gp::PLSPosterior, t::Real) = d2m(gp, [t]) |> arr2sc
d3m(gp::PLSPosterior, t::Real) = d3m(gp, [t]) |> arr2sc
V(gp::PLSPosterior, t::Real) = Σ(gp, [t]) |> arr2sc
Vd(gp::PLSPosterior, t::Real) = Σd(gp, [t]) |> arr2sc
dVd(gp::PLSPosterior, t::Real) = dΣd(gp, [t]) |> arr2sc
V0f(gp::PLSPosterior, t::Real) = Σ0f(gp, [t]) |> arr2sc
Vd0f(gp::PLSPosterior, t::Real) = Σd0f(gp, [t]) |> arr2sc
V0df(gp::PLSPosterior, t::Real) = Σ0df(gp, [t]) |> arr2sc
Vd0df(gp::PLSPosterior, t::Real) = Σd0df(gp, [t]) |> arr2sc


# Update
function update(gp::PLSPosterior{T}, x::PLSEvaluation{T}; update_llh_var::Bool=false) where T <: AbstractFloat
    # Currently this just rebuilds from scratch -- my belief is that
    # this is sufficiently fast, especially the 2-dimensional broadcast on the
    # kernel matrix constructions that there is no need to utilise intermediate
    # results. However, it's an obvious place to look for optimisation if reqd,
    # esp. if you increase the evaluation limit.
    @unpack σ²_f, σ²_∇ = gp
    @unpack t, y, ∇y = x
    N = length(gp) + 1
    Ts, Y, ∇Y = vcat(gp.Ts, t), vcat(gp.Y, y), vcat(gp.∇Y, ∇y)
    update_llh_var && begin; σ²_f, σ²_∇ = (x.σ²_f + (N-1) * σ²_f)/N, (x.σ²_∇ + (N-1) * σ²_∇)/N; end
    return PLSPosterior(Ts, Y, ∇Y, σ²_f, σ²_∇)
end

# Find the minimum mean value over specified Ts (default: current evaluations)
min_mean(x::PLSPosterior) = min_mean(x, x.Ts)
function min_mean(x::PLSPosterior{T}, Ts::Vector{T}) where T <: AbstractFloat
    M  = m(x, Ts)
    minj = argmin(M)
    return M[minj], Ts[minj], minj
end

# Standard methods
Base.length(x::PLSPosterior) = length(x.Ts)
Base.eltype(x::PLSPosterior{T}) where T <: AbstractFloat = T

# allow broadcasting over posterior objects (interpret as scalar)
Broadcast.broadcastable(x::PLSPosterior) = Ref(x)
