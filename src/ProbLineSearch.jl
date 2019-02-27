module ProbLineSearch

using Flux, LinearAlgebra
using Distributions
using ArgCheck, Parameters

# Original code (MATLAB) license:
#
# Copyright (c) 2015 (post NIPS 2015 release 4.0), Maren Mahsereci, Philipp Hennig
# mmahsereci@tue.mpg.de, phennig@tue.mpg.de
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

const ζ = 10     # numerical stability in kernel function (any positive const is ok)


"""
Statistics of accepted step sizes and number of func evals
"""
mutable struct PLSTermination{T<: AbstractFloat}
    accepted_αs::Vector{T} # list of accepted step sizes
    evals::Vector{T}       # number of functions evals
    EI::T                  # Expected Improvement at acceptance
    Wolfe::T               # Wolfe probability at acceptance
    reason::String         # Description of acceptance reason
end
PLSTermination(T::Type, N::Int) = PLSTermination(fill(convert(T, NaN), N), fill(convert(T, NaN), N), 0, 0, "")


"""
PLS constants (see Mahsereci & Hennig, 2017), which shouldn't need to change,
but are available here just in case.
"""
@with_kw struct PLSConstantParams{T<:AbstractFloat}
    verbosity::Int = 1
    c1::T = 0.05  # > pct of ∇₀ decrease required
    c2::T = 0.5   # grad at accepted point must be above this pct of ∇₀
    niter::Int = 7
    wolfeThresh::T = 0.3   # ≥ this prob reqd, for Wolfe conditions to hold
    αGrowth::T = 1.3       # amount to grow α₀ at each outer iteration.
end



"""
Definition of the line search: the starting point, the direction etc.
"""
@with_kw mutable struct PLSSearchDefn{T<:AbstractFloat}
    x₀::Vector{T}
    α₀::T
    search_direction::Vector{T}
    extrap_amt::T = 1      # extrapolation amount
    ewma_α::T              # (exp. wgt) moving average of α (across outer loops)
    m::T = 0               # magnitude |y′(0)| => Divisor for normalisation
    f₀::T = 0
end


"""
Output of a function evaluation.
"""
struct PLSEvaluation{T <: AbstractFloat}
    t::T       # evaluated `t`
    x::T       # corr. x(t)
    y::T       # corr. y(t)
    ∇y::T      # corr. y′(t)
    σ²_f::T    # estimated var of f
    σ²_∇::T    # estimate var of f′
    J::Matrix{T}    # Jacobian at t
    σ²_J::Matrix{T} # Variance of Jacobian at t
end

Base.eltype(x::PLSEvaluation{T}) = T
"""
`AbstractPLSFunction`
Abstract type for function evaluations, which should support the following
interface:

1. [`evaluate`](f::AbstractPLSFunction, t) returns:
       PLSEvaluation(t, f(x), ∇f(x), var f(x), var ∇f(x))

It is very difficult to write a general purpose piece of code that will
result in variance estimates that runs efficiently. I have left this to
individual users, excepting the one type `PLSBespokeFunction` which assumes
the function call returns the values [f(x), ∇f(x), var f(x), var ∇f(x)]
and very little additional work is required (i.e. gradients, and variances).
"""
abstract type AbstractPLSFunction end

struct PLSBespokeFunction <: AbstractPLSFunction
    f::Function
    pars::Any
end

#= TODO: add use of optimiser to change search direction, e.g. ADAM/Adadelta/Nesterov etc. =#

function evaluate(f::PLSBespokeFunction{T}, t::T,
        searchpars::PLSSearchDefn{T}) where T <:AbstractFloat
    @unpack x₀ ,α₀, search_direction, m, f₀ = searchpars
    @unpack f, pars = f
    query = x₀ + α₀*t*search_direction
    ft, J, σ²_f, σ²_J = f(query, pars...)
    ∇f, σ²_∇ = search_direction' * J, (search_direction.^2)' * σ²_J  # project
    ft, ∇f = normalise_for_linesearch(ft, ∇f, f₀, m, α₀)
    return PLSEvaluation(t, query, ft, ∇f, σ²_f, σ²_∇, J, σ²_J)
end


@inline normalise_for_linesearch(f, ∇f, f₀, m, α₀) = (ft - f₀)/(α₀*m),  ∇f/m
@inline zT(N...) = zeros(T, N...)


include("./posteriorgp.jl")


function probLineSearch(func::AbstractPLSFunction, search::PLSSearchDefn, pars::PLSConstantParams)
    x0 = evaluate(func, 0, search)
    state = PLSTermination(eltype(x0), pars.niter))
    state.evals[1] = 1
    probLineSearch(func, x0, search, pars, state=state)
end


function probLineSearch(func::AbstractPLSFunction, x0::PLSEvaluation, search::PLSSearchDefn, pars::PLSConstantParams;
        state::PLSTermination=PLSTermination(eltype(x0), pars.niter))

    @unpack α₀, search_direction = search
    @unpack niter, c1, c2, wolfeThresh, verbosity = pars

    m    = norm(x0.∇y)

    # #####  => I'm not really sure that these are used. ######
        σ_f  = sqrt(x0.σ²_f)/(α₀ * m)
        σ_∇  = sqrt(σ²_∇)/m
    # #####  ........................................... ######

    search.norm_y0′ = m   # magnitude for rescaling: |y′(0)|
    gp = PLSPosterior(zT(1), zT(1), [x0.∇y/m], x0.σ²_f, x0.σ²_∇) # T, Y, ∇Y, ΣY, Σ∇
    pls_stats = PLSTermination(T, niter)

    tt = 1.0   # initial step size (rescaled)

    # =========================================================
    for i = 1:niter
        x = evaluate(func, tt, search)
        gp = update!(gp, tt, x.y, x.∇y)

        # -- check last evaluated point for acceptance ------------------------
        if probWolfe(tt) > wolfeThresh   # are we done?
            state.reason = "found acceptable point."
            # !!!!!! THIS IS PROBABLY WRONG => WORK ALWAYS ASSUMES ACCEPTED PT WILL BE THE LAST EVALUATED !!!!!
            finalise(x, searchpars, pars, state);
            return  # done  => x.x, x.y, x.J, x.∇_J, searchpars.α₀, state.ewma_α
        end


        # -- find candidates (set up for EI) ----------------------------------
        #    get lowest mean of existing


        # -- check this point as well for acceptance --------------------------


        # -- find candidates (1) ----------------------------------------------
        # CANDIDATES 1: minimal means between all evaluations:


        # -- check if at least on point is acceptable -------------------------


        # -- find candidates (2) ----------------------------------------------
        # CANDIDATES 2: one extrapolation step


        # -- discriminate candidates through EI and prob Wolfe ----------------


    end
    # =========================================================
    # => Not found an acceptable candidate in the niter limit.
end



function evaluate(f::PLSBespokeFunction{T}, t::T, searchpars::PLSSearchDefn{T}) where T <:AbstractFloat
    @unpack x₀ ,α₀, search_direction = searchpars
    @unpack f, pars = f
    query = x₀ + α₀*t*search_direction
    f, J, σ²_f, σ²_J = f(query, pars...)
    ∇f, σ²_∇ = search_direction' * J, (search_direction.^2)' * σ²_J
    return PLSEvaluation(t, f, ∇f, σ²_f, σ²_∇)
end


function finalise(x::PLSEvaluation{T}, searchpars::PLSSearchDefn{T},
        pars::PLSConstantParams) where T <: AbstractFloat
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
    searchpars.ewma_α = gamma*searchpars.ewma_α + (1-gamma)*tt*α₀;

    # reset NEXT initial step size to average step size if accepted step
    # size is 100 times smaller or larger than average step size
    if (searchpars.α₀ > 1e2 * searchpars.ewma_α)||(searchpars.α₀ < 1e-2 * searchpars.ewma_α)
        (verbosity > 0) && println('making a very small step, resetting alpha0')
        searchpars.α₀ = searchpars.ewma_α # reset step size
    end
end

end # module
