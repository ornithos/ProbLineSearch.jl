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
probWolfe(Post::PLSPosterior, t, c1, c2) = probWolfe_(Post::PLSPosterior, t, c1, c2)[1]

# probWolfe AND breakdown thereof
function probWolfe_(Post::PLSPosterior, t, c1, c2)

    T = eltype(Post)

    # marginal for Armijo condition
    dm0  = d1m(Post, 0)  # 1st deriv. of mean at t=0
    Vd0  = Vd(Post, 0)   # marginal cov btwn f and f′ at t=0
    dVd0 = dVd(Post, 0)  # marginal cov of f′ at t=0
    ma  = m(Post, 0) - m(Post, t) + c1 * t * dm0;
    Vaa = V(Post, 0) + (c1 * t).^2 * dVd0 + V(Post, t) + 2 * (c1 * t * (Vd0 - Vd0f(Post, t)) - V0f(Post, t));

    # marginal for curvature condition
    mb  = d1m(Post, t) - c2 * dm0;
    Vbb = c2^2 * dVd0 - 2 * c2 * Vd0df(Post, t) + dVd(Post, t);

    # covariance between conditions
    Vab = -c2 * (Vd0 + c1 * t * dVd0) + V0df(Post, t) + c2 * Vd0f(Post, t) + c1 * t * Vd0df(Post, t) - Vd(Post, t);

    if (Vaa < 1e-9) && (Vbb < 1e-9) # ≈ deterministic evaluations, returns to standard Wolfe conditions.
        return (ma >= 0) .* (mb >= 0), zT(T, 3)
    end

    # joint probability
    rho = Vab / sqrt(Vaa * Vbb);
    if Vaa <= 0 || Vbb <= 0
        return 0, zT(eltype(Post), 3)
    end
    upper = (2 * c2 * (abs(dm0)+2*sqrt(dVd0))-mb)./sqrt(Vbb);
    p = bvn(-ma / sqrt(Vaa), convert(T, Inf), -mb / sqrt(Vbb), upper, rho);

    # individual marginal probabilities for each condition
    # (for debugging)
    p12 = [1 - gaussCDF(-ma/sqrt(Vaa)), gaussCDF(upper)-gaussCDF(-mb/sqrt(Vbb)),
            Vab / sqrt(Vaa * Vbb)];

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
        return - (d1mt - ts * d2mt) / d2mt;
    end

    # compute the two possible roots:
    detmnt = b^2 - 4*a*c;
    if detmnt < 0 # no roots
        return convert(T, Inf);
    end
    LR = (-b - sign(a) * sqrt(detmnt)) / (2*a);  # left root
    RR = (-b + sign(a) * sqrt(detmnt)) / (2*a);  # right root

    # and the two values of the cubic at those points (up to constant)
    Ldt = LR - ts; # delta t for left root
    Rdt = RR - ts; # delta t for right root
    LCV = d1mt * Ldt + 2 \ d2mt * Ldt.^2 + 6 \ d3mt * Ldt.^3; # left cubic value
    RCV = d1mt * Rdt + 2 \ d2mt * Rdt.^2 + 6 \ d3mt * Rdt.^3; # right cubic value

    return (LCV < RCV) ? LR : RR
end
