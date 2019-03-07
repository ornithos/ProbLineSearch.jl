
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
Base.copy(x::PLSEvaluation) = PLSEvaluation([deepcopy(getfield(x, nm)) for nm in fieldnames(PLSEvaluation)]...)


@inline normalise_for_linesearch(f, ∇f, f₀, norm∇_0, α₀) = (f - f₀)/(α₀*norm∇_0),  ∇f/norm∇_0
@inline invert_normalise_for_linesearch(f, ∇f, f₀, norm∇_0, α₀) = f*(α₀*norm∇_0) + f₀,  ∇f*norm∇_0

function normalise_for_linesearch!(x::PLSEvaluation{T}, searchpars::PLSSearchDefn{T}) where T <: AbstractFloat
    x_m = norm(x.∇y)
    isapprox(x_m, 1.0) && return   # assume already been done elsewhere
    @unpack α₀, f₀, denom = searchpars

    y, ∇y = normalise_for_linesearch(x.y, x.∇y, f₀, denom, α₀)
    σ²_f, σ²_∇ = normalise_for_linesearch(x.σ²_f, x.σ²_∇, T(0), denom^2, α₀^2)
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


function normalised_fun_var(searchpars::PLSSearchDefn{T}) where T <: AbstractFloat
    @unpack α₀, f₀, denom, σ²_f = searchpars
    σ²_∇ = var∇1d(searchpars)
    normalise_for_linesearch(σ²_f, σ²_∇, T(0), denom^2, α₀^2)
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
    pars::Dict
end
PLSBespokeFunction(f::Function) = PLSBespokeFunction(f, Dict())


#= TODO: splatting a kwargs instead of args adds ≈1μs per call. Check at some point
         what the overall runtime is, and whether it is worth optimising this. =#

#= TODO: add use of optimiser to change search direction, e.g. ADAM/Adadelta/Nesterov etc. =#


function evaluate(f::PLSBespokeFunction, t::T, searchpars::PLSSearchDefn{T};
        history::Union{PLSHistory{T}, Nothing}=nothing, normalise::Bool=true) where T <:AbstractFloat
    @unpack x₀ ,α₀, search_direction = searchpars
    query = x₀ + α₀*t*search_direction
    # @show query
    x = evaluate(f, query, history=history)
    x.t = t
    project_1d!(x, search_direction)
    normalise && normalise_for_linesearch!(x, searchpars)
    return x
end


function evaluate(f::PLSBespokeFunction, query::Array{T};
        history::Union{PLSHistory{T}, Nothing}=nothing) where T <:AbstractFloat
    @unpack f, pars = f
    ft, J, σ²_f, σ²_J = f(query; pars...)
    x = PLSEvaluation{T}(T(NaN), query, ft, T(NaN), σ²_f, T(NaN), J, σ²_J)
    !(history === nothing) && increment_eval!(history)
    return x
end


function (f::PLSBespokeFunction)(x; gradient=false)
    y, J, σ²_y, σ²_J = f.f(x; f.pars...)
    return gradient ? (y, J) : y
end


# function estimate_variance(f::PLSBespokeFunction, x::Array{T}) where T <: AbstractFloat
#     y, J, σ²_y, σ²_J = f.f(x; f.pars...)
#     return σ²_y, σ²_J
# end


function project_1d!(x::PLSEvaluation, search_direction)
    x.∇y = search_direction' * x.J
    x.σ²_∇ = (search_direction.^2)' * x.σ²_J
end

function Base.show(io::IO, x::PLSEvaluation)
    println(io, "ProbLineSearch evaluation:")
    println(io, "evaluated at $(x.x)")
    print(io, "Y = $(x.y), ∇ = $(x.∇y)")
end
