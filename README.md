# ProbLineSearch
Probabilistic Line Searches for Stochastic Optimization (Mahsereci &amp; Hennig, 2017) ported from MATLAB.

Basic implementation complete. Basic testing suggests that the code is directly equivalent to the MATLAB version. The conversion is not entirely trivial since most operations in original code are performed through mutation of parent scope. Since this version avoids any global or 'multi-scope' variables, performance is typically between 10x-30x faster on the test functions.

### Usage:
The interface is perhaps a little clumsy at present. A minimal working example can be run as follows:

1. **Wrap test function** in `PLS` wrapper. This will eventually be to provide a consistent interface to different function types, to retain state (if reqd) and other function arguments. This may be required for instance if a cyclical minibatch is used and the current batch number must be stored. It may also be useful in order to provide estimates of the variance of function and gradient evaluations. The eventual goal is to provide a wrapper for Automatic Differentiation ([Flux](https://github.com/FluxML/Flux.jl) etc.) for this, but it may not be easy to provide an efficient implementation in general. For the time we use a function wrapper `PLSBespokeFunction` which assumes a bespoke implementation of a chosen function which outputs (`function value` f, `gradient estimate` ∇, `variance estimate for` f, `variance estimate for` ∇):

**Test Function**
```julia
function mccormick2(x::Vector{T}; σ_f = 0.5, σ_df=0.8) where T <: AbstractFloat

    a = sin(sum(x));
    b = cos(sum(x));

    f =   a + (x[1] - x[2]).^2 - 1.5.*x[1] + 2.5.* x[2] + 1;
    df = [b + 2*(x[1] - x[2]) - 1.5, b - 2*(x[1] - x[2]) + 2.5];

    return f + randn(T)*σ_f, df + randn(T,2)*σ_df, σ_f^2, σ_df^2 * ones(T,2)
end
```

**Wrapper**
```julia
plsfun = PLSBespokeFunction(mccormick2, Dict(:σ_f=>1.0, :σ_df=>1.0))
```

2. **Initialise line search from a x₀**

There's a useful initialisation signature which provides easier access to the various objects the probLineSearch procedure needs to carry around. Each call of the `probLineSearch` performs a single line search for a given direction. In particular, the `PLSSearchDefn` defines the initial position, search direction, initial step size (among other properties) which allows the `probLineSearch` procedure to pretend that the function in question is 1D. Other objects of interest include the `PLSHistory` which captures the accepted step sizes and associated messages, as well as a moving average thereof, and a `PLSPosterior`, the Gaussian Process (with Integrated Wiener kernel) which captures the evaluations and uncertainty of the search. The latter is currently returned for debug and visualisation purposes (TODO: this should be avoided, or wrapped with all other return values.)
```julia
x, gp, search, history = probLineSearch(plsfun, x₀, 0.01)
```
Here the initial x₀ is a vector, and the third argument is the initial stepsize (α₀). A `search_direction` can optionally be specified as a keyword argument, otherwise the procedure evaluates the gradient at x₀ and uses this. Note that the return value `x` is not simply a vector, but a `PLSEvaluation` which contains the position `x.x` as well as the function value `x.y`, gradient of previous line search `x.∇y`, Jacobian `x.J` and the estimates of the variance of evaluations and gradients at `x.x`.

3. **Iterate the line search using a stochastic optimisation algorithm**

The simplest option is to use Stochastic Gradient Descent (SGD) which simply uses the `x.J` from the previous iteration as the (negative) search direction. An example of performing basic SGD is as follows:

```julia
xs, gps = [], []               # save iterates and line search posteriors for debug
for i in 1:niter
    search.search_direction = -x.J                # choose SGD search direction
    x, gp = probLineSearch(plsfun, x, search, history=history)
    push!(xs, copy(x))
    push!(gps, gp)
end
```
Of course more interesting search directions may be specified. The history object is modified in place, and it is important to keep passing the same object through. The moving average of α is used in order to determine if the current step size is unusually high or low, and if not passed in, the optimisation can get stuck by deeming the chosen α inadmissable.

4. **Line Search parameters**
The parameters hard-coded into the original implementation have been moved into another object, which allows specification of `verbosity`, `c1`, `c2` (Wolfe conditions), `wolfeThresh` (probability that the Wolfe conditions hold), `niter`, the maximum number of evaluations within each line search, and `αGrowth`, the multiplier of α after every iteration. All of these parameters can be passed into the line search via e.g. `pars=PLSConstantParams(verbosity=2, niter=10)`.

## To do
This is an initial implementation purely for my own research. I intend to tidy this up a lot, but I may need prodding, so please give me a shout if you're interested in using this, and want a less ugly version.

## Additional notes
This implementation includes a Julia translation of the `BVN` routine written by [Alan Genz](http://www.math.wsu.edu/faculty/genz/homepage). This code calculates rectangular integrals of bivariate unit Gaussians with arbitrary correlation. The code has been extensively tested against the MATLAB version and appears to be correct. The code also runs about an order of magnitude faster than the original, and perhaps can be improved still.
