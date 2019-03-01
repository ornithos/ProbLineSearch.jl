
"""
Output of a function evaluation.
"""
mutable struct PLSEvaluation{T <: AbstractFloat}
    t::T       # evaluated `t`
    x::Vector{T}       # corr. x(t)
    y::T       # corr. y(t)
    ∇y::T      # corr. y′(t)
    σ²_f::T    # estimated var of f
    σ²_∇::T    # estimate var of f′
    J::Vector{T}    # Jacobian at t
    σ²_J::Vector{T} # Variance of Jacobian at t
end

Base.eltype(x::PLSEvaluation{T}) where T <: AbstractFloat = T


@inline normalise_for_linesearch(f, ∇f, f₀, norm∇_0, α₀) = (f - f₀)/(α₀*norm∇_0),  ∇f/norm∇_0
@inline invert_normalise_for_linesearch(f, ∇f, f₀, norm∇_0, α₀) = f*(α₀*norm∇_0) + f₀,  ∇f*norm∇_0

function normalise_for_linesearch!(x::PLSEvaluation{T}, searchpars::PLSSearchDefn{T}) where T <: AbstractFloat
    x_m = norm(x.∇y)
    isapprox(x_m, 1.0) && return   # assume already been done elsewhere
    @unpack α₀, f₀, denom = searchpars

    # TODO: THIS FEELS WRONG TO ME. WHY NORMALISE *Y* BY α? SURELY THIS IS JUST
    #=       A TRANSFORMATION FOR X, TO NORMALISE RANGE TO [0,1]? BY NORMALISING
    #       BY α, THIS DRAMATICALLY CHANGES the EMISSION NOISE. OF COURSE IF
    #       x <-- x/α, then ∇y <-- ∇y / α, but *[NOT y <-- y/α]*
    =#
    y, ∇y = normalise_for_linesearch(x.y, x.∇y, f₀, denom, α₀)
    σ²_f, σ²_∇ = normalise_for_linesearch(x.σ²_f, x.σ²_∇, T(0), denom^2, α₀)
    x.y, x.∇y = y, ∇y
    x.σ²_f, x.σ²_∇ = σ²_f, σ²_∇
end

function invert_normalise_for_linesearch!(x::PLSEvaluation{T}, searchpars::PLSSearchDefn{T}) where T <: AbstractFloat
    x_m = norm(x.∇y)
    isapprox(x_m, searchpars.denom) && begin; @warn "invert_normalise not necessary"; return; end
    @unpack α₀, f₀, denom = searchpars
    y, ∇y = invert_normalise_for_linesearch(x.y, x.∇y, f₀, denom, α₀)
    σ²_f, σ²_∇ = invert_normalise_for_linesearch(x.σ²_f, x.σ²_∇, T(0), denom^2, α₀^2)
    x.y, x.∇y = y, ∇y
    x.σ²_f, x.σ²_∇ = σ²_f, σ²_∇
end

"""
`AbstractPLSFunction`
Abstract type for function evaluations, which should support the following
interface:

1. [`evaluate`](f::AbstractPLSFunction, t) returns:
       PLSEvaluation(f(x), ∇f(x), var f(x), var ∇f(x))

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

function evaluate(f::PLSBespokeFunction, t::T, searchpars::PLSSearchDefn{T};
        history::Union{PLSHistory{T}, Nothing}=nothing, normalise::Bool=true) where T <:AbstractFloat
    @unpack x₀ ,α₀, search_direction = searchpars
    @unpack f, pars = f
    query = x₀ + α₀*t*search_direction
    ft, J, σ²_f, σ²_J = f(query, pars...)
    # Project onto 1D line (search direction), and renormalise.
    ∇f, σ²_∇ = search_direction' * J, (search_direction.^2)' * σ²_J  # project
    x = PLSEvaluation{T}(t, query, ft, ∇f, σ²_f, σ²_∇, J, σ²_J)
    normalise && normalise_for_linesearch!(x, searchpars)
    !(history === nothing) && increment_eval!(history)
    return x
end

function (f::PLSBespokeFunction)(x; gradient=false)
    y, J, σ²_y, σ²_J = f.f(x, f.pars)
    return gradient ? (y, J) : y
end

function Base.show(io::IO, x::PLSEvaluation)
    println(io, "ProbLineSearch evaluation:")
    println(io, "evaluated at $(x.x)")
    print(io, "Y = $(x.y), ∇ = $(x.∇y)")
end
