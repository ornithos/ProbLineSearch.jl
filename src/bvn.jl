"""
This is a Julia port of the MATLAB routine ``BVN'' by Alan Genz.
It has been fairly comprehensively tested against the results of the original
in MATLAB, and runs currently close to 10x faster.

Translated by Alex Bird.
"""

function gaussCDF(z::T)::T where T <: AbstractFloat
    cdf(Normal(zero(T), one(T)), z)
end

function bvn(xl::Real, xu::Real, yl::Real, yu::Real, r::Real)
    xl, xu, yl, yu, r = promote(xl, xu, yl, yu, r)
    bvn(xl, xu, yl, yu, r)
end


function bvn(xl::T, xu::T, yl::T, yu::T, r::T) where T <: AbstractFloat
# BVN
#  A function for computing bivariate normal probabilities.
#  bvn calculates the probability that
#    xl < x < xu and yl < y < yu,
#  with correlation coefficient r.
#   p = bvn( xl, xu, yl, yu, r )
#
#
#   Author
#       Alan Genz, Department of Mathematics
#       Washington State University, Pullman, Wa 99164-3113
#       Email : alangenz@wsu.edu

    p = bvnu(xl,yl,r) - bvnu(xu,yl,r) - bvnu(xl,yu,r) + bvnu(xu,yu,r);
    return clamp(p, 0, 1)
end


function bvnu(dh::T, dk::T, r::T)::T where T <: AbstractFloat
# BVNU
#  A function for computing bivariate normal probabilities.
#  bvnu calculates the probability that x > dh and y > dk.
#    parameters
#      dh 1st lower integration limit
#      dk 2nd lower integration limit
#      r   correlation coefficient
#  Example: p = bvnu( -3, -1, .35 )
#  Note: to compute the probability that x < dh and y < dk,
#        use bvnu( -dh, -dk, r ).
#
#
#   Author
#       Alan Genz
#       Department of Mathematics
#       Washington State University
#       Pullman, Wa 99164-3113
#       Email : alangenz@wsu.edu
#
#    This function is based on the method described by
#        Drezner, Z and G.O. Wesolowsky, (1989),
#        On the computation of the bivariate normal inegral,
#        Journal of Statist. Comput. Simul. 35, pp. 101-107,
#    with major modifications for double precision, for |r| close to 1,
#    and for Matlab by Alan Genz. Minor bug modifications 7/98, 2/10.
#
#
# Copyright (C) 2013, Alan Genz,  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided the following conditions are met:
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#   3. The contributor name(s) may not be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    # deal with easy cases first (if one or other limit is ∞ or indpdt)
    (dh == Inf || dk == Inf) && return 0
    (dh == -Inf) && return (dk == -Inf) ? 1 : gaussCDF(-dk);
    (dk == -Inf) && return gaussCDF(-dh);
    (r == 0) && return gaussCDF(-dh)*gaussCDF(-dk);

    # now we have that the bounds are finite, and corrcoeff is !== 0.
    tp = 2*pi; h = dh; k = dk; hk = h*k; bvn = 0.0;

    # set up Gauss Legendre points / weights for integration
    if abs(r) < 0.3      # n =  6
        w = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904];
        x = [0.9324695142031522, 0.6612093864662647, 0.2386191860831970];
    elseif abs(r) < 0.75 # n = 12
        w = [.04717533638651177, 0.1069393259953183, 0.1600783285433464,
             0.2031674267230659, 0.2334925365383547, 0.2491470458134029];
        x = [0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
             0.5873179542866171, 0.3678314989981802, 0.1252334085114692];
    else                # n = 20
        w = [.01761400713915212, .04060142980038694, .06267204833410906,
             .08327674157670475, 0.1019301198172404, 0.1181945319615184,
             0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
             0.1527533871307259];
        x = [0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
             0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
             0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
             0.07652652113349733];
    end

    w, x = convert.(T, vcat(w, w)), convert.(T, vcat(1.0 .- x, 1.0 .+ x));

    # easier case when correlation is not too large.
    if abs(r) < 0.925
        hs  = (h^2 + k^2)/2;
        asr = asin(r)/2;
        sn  = sin.(asr*x);
        bvn = exp.((sn*hk .- hs)./(1.0 .- sn.^2))'*w;
        bvn = bvn * asr/tp + gaussCDF(-h)*gaussCDF(-k);
    else
        if r < 0
            k, hk = -k, -hk
        end
        if abs(r) < 1
            as = 1.0-r^2;
            a = sqrt(as);
            bs = (h-k)^2;
            asr = -( bs/as + hk )/2;
            c = (4.0-hk)/8.0;
            d = (12.0-hk)/80.0;
            if asr > -100;
                bvn = a*exp(asr)*(1-c*(bs-as)*(1-d*bs)/3+c*d*as^2);
            end
            if hk  > -100;
                b = sqrt(bs);
                sp = sqrt(tp)*gaussCDF(-b/a);
                bvn -= exp(-hk/2)*sp*b*( 1 - c*bs*(1-d*bs)/3 );
            end
            a = a/2;
            xs = (a*x).^2;
            asr = -( bs./xs .+ hk )/2;
            ix = findall( asr .> -100.0 );
            xs = xs[ix];
            sp = ( 1.0 .+ c*xs.*(1.0 .+ 5.0*d*xs) );
            rs = sqrt.(1.0 .- xs);
            ep = exp.( -(hk/2)*xs./(1.0 .+ rs).^2 )./rs;
            bvn = ( a*( (exp.(asr[ix]).*(sp-ep))'*w[ix] ) - bvn )/tp;
        end
        if r > 0
            bvn =  bvn + gaussCDF( -max( h, k ) );
        elseif h >= k
            bvn = -bvn;
        else
            if h < 0
                L = gaussCDF(k)-gaussCDF(h);
            else
                L = gaussCDF(-h)-gaussCDF(-k);
            end
            bvn =  L - bvn;
        end
    end
    p = clamp(bvn, 0.0, 1.0)
    return p
end
