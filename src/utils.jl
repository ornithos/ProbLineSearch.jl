include("./bvn.jl")


function gaussPDF(z::T)::T where T <: AbstractFloat
    pdf(Normal(zero(T), one(T)), z)
end

# Expected improvement:
# m = mean, s = std dev, η = current best
function EI(m,s,η)
    return (η - m) .* gaussCDF((η-m)./s) + s .* gaussPDF((η-m)./s);
end

# probability for Wolfe conditions to be fulfilled
probWolfe(Post::PLSPosterior{T}, t::T, c1::T, c2::T) where T <: AbstractFloat = probWolfe_(Post::PLSPosterior, t, c1, c2)[1]


# For lower precision arithmetic, we sometimes encounter small negative numbers
# tolerant_sqrt(x) = x < 0 ? 0 : sqrt(x)
__sqrt(x::Float64) = (-1e-9 < x < 0) ? 0 : sqrt(x)
__sqrt(x::Float32) = (-1e-5 < x < 0) ? 0 : sqrt(x)
__sqrt(x::Float16) = (-1e-1 < x < 0) ? 0 : sqrt(x)

# probWolfe AND breakdown thereof
function probWolfe_(Post::PLSPosterior{T}, t::T, c1::T, c2::T)  where T <: AbstractFloat

    T0 = T(0)
    # marginal for Armijo condition
    dm0  = d1m(Post, T0)  # 1st deriv. of mean at t=0
    Vd0  = Vd(Post, T0)   # marginal cov btwn f and f′ at t=0
    dVd0 = dVd(Post, T0)  # marginal cov of f′ at t=0
    ma  = m(Post, T0) - m(Post, t) + c1 * t * dm0;
    Vaa = V(Post, T0) + (c1 * t).^2 * dVd0 + V(Post, t) + 2 * (c1 * t * (Vd0 - Vd0f(Post, t)) - V0f(Post, t));

    # marginal for curvature condition
    mb  = d1m(Post, t) - c2 * dm0;
    Vbb = c2^2 * dVd0 - 2 * c2 * Vd0df(Post, t) + dVd(Post, t);

    # covariance between conditions
    Vab = -c2 * (Vd0 + c1 * t * dVd0) + V0df(Post, t) + c2 * Vd0f(Post, t) + c1 * t * Vd0df(Post, t) - Vd(Post, t);

    if (Vaa < 1e-9) && (Vbb < 1e-9) # ≈ deterministic evaluations, returns to standard Wolfe conditions.
        return (ma >= 0) .* (mb >= 0), zeros(T, 3)
    end

    # joint probability
    rho = Vab / __sqrt(Vaa * Vbb);
    if Vaa <= 0 || Vbb <= 0
        return T(0), zeros(T, 3)
    end
    upper = (2 * c2 * (abs(dm0)+2*__sqrt(dVd0))-mb)./__sqrt(Vbb);
    p = bvn(-ma / __sqrt(Vaa), convert(T, Inf), -mb / __sqrt(Vbb), upper, rho);

    # individual marginal probabilities for each condition
    # (for debugging)
    p12 = [1 - gaussCDF(-ma/__sqrt(Vaa)), gaussCDF(upper)-gaussCDF(-mb/__sqrt(Vbb)),
            Vab / __sqrt(Vaa * Vbb)];

    return p, p12
end



function cubicMinimum(Post::PLSPosterior{T}, t::T) where T <: AbstractFloat
    # mean belief at ts is a cubic function. It is defined up to a constant by
    d1mt = d1m(Post, t);
    d2mt = d2m(Post, t);
    d3mt = d3m(Post, t);

    a = d3mt / 2;
    b = d2mt - t * d3mt;
    c = d1mt - d2mt * t + 2 \ d3mt * t^2;

    if abs(d3mt) < 1e-9 # essentially a quadratic. Single extremum
        return - (d1mt - t * d2mt) / d2mt;
    end

    # compute the two possible roots:
    detmnt = b^2 - 4*a*c;
    if detmnt < 0 # no roots
        return convert(T, Inf);
    end
    LR = (-b - sign(a) * __sqrt(detmnt)) / (2*a);  # left root
    RR = (-b + sign(a) * __sqrt(detmnt)) / (2*a);  # right root

    # and the two values of the cubic at those points (up to constant)
    Ldt = LR - t; # delta t for left root
    Rdt = RR - t; # delta t for right root
    LCV = d1mt * Ldt + 2 \ d2mt * Ldt.^2 + 6 \ d3mt * Ldt.^3; # left cubic value
    RCV = d1mt * Rdt + 2 \ d2mt * Rdt.^2 + 6 \ d3mt * Rdt.^3; # right cubic value

    return (LCV < RCV) ? LR : RR
end


@inline arr2sc(x::AbstractArray) = begin; @argcheck length(x)==1; x[1]; end



# Online variance calculation using Welford's algorithm
mutable struct OnlineVarianceVec{T<:AbstractFloat}
    n::Int
    μ::Vector{T}
    M2::Vector{T}
end

OnlineVarianceVec(val::Vector{T}) where T <: AbstractFloat = OnlineVarianceVec{T}(1, val, zeros(T, length(val)))
OnlineVarianceVecInit(T::Type, N::Int) = OnlineVarianceVec{T}(0, zeros(T, N), zeros(T,N))

function update!(x::OnlineVarianceVec{T}, val::Vector{T}) where T <: AbstractFloat
    x.n += 1
    Δ₁ = val - x.μ
    x.μ .+= Δ₁ ./ x.n
    Δ₂ = val - x.μ
    x.M2 .+= Δ₁ .* Δ₂
end


value(x::OnlineVarianceVec{T}) where T <: AbstractFloat = (x.n < 2) ? T(NaN) : x.M2 ./ (x.n - 1)
