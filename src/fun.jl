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

function evaluate(f::PLSBespokeFunction{T}, t::T, searchpars::PLSSearchDefn{T}
        history::PLSTermination{T}) where T <:AbstractFloat
    @unpack x₀ ,α₀, search_direction, m, f₀ = searchpars
    @unpack f, pars = f
    query = x₀ + α₀*t*search_direction
    ft, J, σ²_f, σ²_J = f(query, pars...)
    ∇f, σ²_∇ = search_direction' * J, (search_direction.^2)' * σ²_J  # project
    ft, ∇f = normalise_for_linesearch(ft, ∇f, f₀, m, α₀)
    history.evals += 1
    return PLSEvaluation(t, query, ft, ∇f, σ²_f, σ²_∇, J, σ²_J)
end
