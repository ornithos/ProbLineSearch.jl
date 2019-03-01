module ProbLineSearch

using LinearAlgebra
using Distributions
using ArgCheck, Parameters, Formatting

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


"""
Statistics of accepted step sizes and number of func evals
example usage: PLSHistory(Float32, 10)
"""
# TODO: mv PLSHistory => PLSStatistics, make
# TODO: cat PLSHistory = {all_evals, all_EI, all_wolfe, all_messages}
@with_kw mutable struct PLSHistory{T <: AbstractFloat}
    evals::Vector{Int} = []       # number of functions evals
    recording::Bool = false            # number of outer iterations so far.
    αs::Vector{T} = []                 # accepted αs
    ewma_α::T = 1.0                    # (exp. wgt) moving average of α (across outer loops)
    EI::Vector{T} = []                 # Expected Improvement at acceptance
    Wolfe::Vector{T} = []              # Wolfe probability at acceptance
    msg::Vector{String} = []   # Description of acceptance reason
end
PLSHistory(T::Type) = PLSHistory{T}()
PLSHistory() = PLSHistory(Float64)

Base.length(x::PLSHistory) = length(x.evals)
increment_eval!(x::PLSHistory) = x.recording ? x.evals[end] += 1 : (push!(x.evals, 1); x.recording=true);


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
PLSConstantParams(T::Type) = PLSConstantParams{T}()


"""
Definition of the line search: the starting point, the direction etc.
"""
@with_kw mutable struct PLSSearchDefn{T<:AbstractFloat}
    x₀::Vector{T}
    α₀::T
    search_direction::Vector{T}
    extrap_amt::T = 1      # extrapolation amount
    denom::T = 1               # magnitude |y′(0)| => Divisor for normalisation
    f₀::T = 0
end
PLSSearchDefn(T::Type, x₀, α₀, search_direction) = PLSSearchDefn{T}(x₀=x₀, α₀=α₀, search_direction=search_direction)
PLSSearchDefn(x₀, α₀, search_direction) = PLSSearchDefn(Float64, x₀, α₀, search_direction)




include("./fun.jl")
include("./posteriorgp.jl")
include("./utils.jl")
# include("./plot.jl")
include("./main.jl")

end # module
