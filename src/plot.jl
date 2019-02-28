using PyPlot
using PyCall
@pyimport matplotlib.colors as matcolors

mpg = [0,0.4717,0.4604]; # color [0,125,122]
dre = [0.4906,0,0]; # color [130,0,0]
ora = [255,153,51] ./ 255;
blu = [0,0,0.509];
gra = 0.5 * ones(3);

lightmpg = [1,1,1] - 0.5 * ([1,1,1] - mpg);
lightdre = [1,1,1] - 0.5 * ([1,1,1] - dre);
lightblu = [1,1,1] - 0.5 * ([1,1,1] - blu);
lightora = [1,1,1] - 0.5 * ([1,1,1] - ora);

ora2white = matcolors.LinearSegmentedColormap[:from_list]("", [
        (0.0,Tuple(ora)),
        (1.0,(1.0, 1.0, 1.0))])

function _GaussDensity(y,m,v)
    exponent = -0.5*(y .- m').^2 ./ v'
    return exp.(exponent) ./ (sqrt(2π) * sqrt(v'))
end


function makePlot(x::PLSEvaluation{T}, post::PLSPosterior{T}, searchpars::PLSSearchDefn{T},
        pars::PLSConstantParams, state::PLSTermination{T}; Tcand=nothing, Mcand=nothing) where T <: AbstractFloat

    @unpack Ts, Y, ∇Y = post
    @unpack wolfeThresh = pars

    if pars.verbosity == 2
        ymin = min(Y) - 0.1*(max(Y)-min(Y));
        ymax = max(Y) + 0.1*(max(Y)-min(Y));

        xmin = -0.1;
        xmax = maximum(vcat(Ts, x.t))+0.5;

        # plot evaluation points
        plot(Ts, Y, 'o', c=blu)
        for i = 1:N
           plot(Ts[i] + 0.1*[-1,1], Y[i] + 0.1*∇Y[i] * [-1,1], '-', c=blu);
        end
        plot(x.t, m(x.t),'o', c=dre, markerfacecolor=dre);

        xlim([xmin,xmax]);
        ylim([ymin,ymax]);
    end

    if pars.verbosity > 2

        doCands = !(Tcand === nothing) && !(Mcand === nothing)  # candidate point plots

        # ----------- PLOT 1 ----------------------
        fig, axs = subplot(3, 2);
        ax = axs[1]
        ax[:title]('belief over function');

        ymin = min(Y) - 1.5*(max(Y)-min(Y));
        ymax = max(Y) + 1.5*(max(Y)-min(Y));

        xmin = -0.1;
        xmax = max(vcat(T, x.t))+0.5;

        # also plot GP
        Np = 120;
        tp = Matrix(collect(range(xmin, stop=xmax, length=Np))');
        ts = unique([tp; Ts[:]]);
        Ns = length(ts);

        mp = m.(post, tp)
        Vp = V.(post, tp)
        ms = m.(post, ts)
        Vs = V.(post, ts)

        PP = zeros(Ns,1);
        Pab = zeros(Ns,3);
        for i = 1:Ns
            p, pab = probWolfe_(ts[i]);
            PP[i] = p; Pab[i,:] = pab;
        end
        yp = Matrix(collect(range(ymin,stop=ymax,length=250))');
        P = _GaussDensity(yp,mp,Vp+1e-4);

        imshow(tp, yp, P, cmap=ora2white);
        plot(ts, ms, '-', c=ora);
        plot(ts, ms + 2*sqrt.(max.(Vs,0)), '-', c=lightora);
        plot(ts, ms - 2*sqrt.(max.(Vs,0)), '-', c=lightora);

        # plot evaluation points
        ax[:plot](Ts, Y,'o',c=blu)
        for i = 1:N
           ax[:plot](Ts[i] + 0.1*[-1,1], Y[i] + 0.1*∇Y[i] * [-1,1], '-', c=blu);
        end
        ax[:plot](x.t, m(post, x.t), 'o', c=dre, markerfacecolor=dre);
        ax[:plot]([x.t, x.t],[ymin,ymax],'-',c=dre);

        doCands && ax[:plot](Tcand, Mcand, 'o', c=gra);

        ax[:xlim]([xmin, xmax]);
        ax[:ylim]([ymin, ymax]);

        # ----------- PLOT 2 ----------------------
        ax = axs[2]
        ax[:title]('belief over Wolfe conditions');
        ax[:plot](ts,PP,'-','Color',dre);
        ax[:plot](ts,Pab(:,1),'--','Color',dre);
        ax[:plot](ts,Pab(:,2),'-.','Color',dre);
        ax[:plot](ts,0.5 + 0.5*Pab(:,3),':','Color',dre);

        ax[:plot](ts, 0*ts + wolfeThresh, '-', c=gra)
        for i = 1:N
            ax[:plot]([Ts[i],Ts[i]],[0,1],'-', c=blu);
        end
        ax[:plot]([x.t, x.t],[0,1],'-',c=dre);
        ax[:ylim]([0,1]);
        ax[:xlim]([xmin,xmax])

        # ----------- PLOT 3 ----------------------
        ax = axs[3]
        ax[:title]('Expected Improvement')

        eta = min(Y);
        Ss  = sqrt(Vs + post.σ²_f);
        ax[:plot](ts, EI.(ms,Ss,eta),'-', c=mpg);
        ax[:plot](ts, EI.(ms,Ss,eta) .* PP,'--', c=blu);

        doCands && [ax[:axvline](tt, c=gra) for tt in Tcand]

        ax[:axvline](x.t, c=dre);
        ax[:xlim]([xmin,xmax]);

        # ----------- PLOT 4 ----------------------
        ax = axs[4]
        ax[:title]('beliefs over derivatives')
        # rhoD = zeros(Ns,1);
        #
        # V0s   = zeros(Ns,1);
        # Vd0s  = zeros(Ns,1);
        # V0ds  = zeros(Ns,1);
        # Vd0ds = zeros(Ns,1);
        #
        # ma  = zeros(Ns,1);
        # mb  = zeros(Ns,1);
        # Vaa = zeros(Ns,1);
        # Vbb = zeros(Ns,1);
        # Vab = zeros(Ns,1);
        #
        # dms = d1m.(post, ts)
        # Vms = dVd.(post, ts)
        #
        # for i = 1:Ns
        #
        #     rhoD(i)  = Vd(ts(i)) ./ sqrt(V(ts(i)) * dVd(ts(i)));
        #     V0s(i)   = V0f(ts(i)) ./ sqrt(V0 * V(ts(i)));
        #     Vd0s(i)  = Vd0f(ts(i))./ sqrt(dVd0 * V(ts(i)));
        #     V0ds(i)  = V0df(ts(i))./ sqrt(V0 * dVd(ts(i)));
        #     Vd0ds(i) = Vd0df(ts(i))./sqrt(dVd0 * dVd(ts(i)));
        #
        #     ma(i)    = m0 - m(ts(i)) + c1 * ts(i) * dm0;
        #     mb(i)    = d1m(ts(i)) - c2 * dm0;
        #     Vaa(i)   = V0 + (c1 * ts(i)).^2 * dVd0 + V(ts(i)) + 2 * (c1 * ts(i) * (Vd0 - Vd0f(ts(i))) - V0f(ts(i)));
        #     Vbb(i)   = c2^2 * dVd0 - 2 * c2 * Vd0df(ts(i)) + dVd(ts(i));
        #     Vab(i)   = -c2 * (Vd0 + c1 * ts(i) * dVd0) + (1+c2) * Vd0f(ts(i)) + c1 * ts(i) * Vd0df(ts(i)) - Vd(ts(i));
        # end
        # plot(ts,dms,'-','Color',ora);
        # plot(ts,dms+2*sqrt.(Vms),'-','Color',lightora);
        # plot(ts,dms-2*sqrt.(Vms),'-','Color',lightora);
        # plot(T,dY_projected,'o','Color',blu);
        #
        # xlim([xmin,xmax]);

        # ----------- PLOT 5 ----------------------
        ax = axs[5]
        ax[:title]('covariances')
        ax[:plot](ts,rhoD,'-', c=ora);
        ax[:plot](ts,V0s,'-', c=mpg);
        ax[:plot](ts,Vd0s,'-.', c=mpg);
        ax[:plot](ts,V0ds,'--', c=mpg);
        ax[:plot](ts,Vd0ds,':', c=mpg);
        ax[:plot](ts,-1+0*ts, '-k');
        ax[:plot](ts,1+0*ts, '-k');

        ax[:xlabel]('t');
        ax[:legend](L"\rho_{f\partial}(t)", L"\rho_{f_0f}(t)",
            L"\rho_{\partial_0f}(t)\rho_{f_0\partial}(t)", L"\rho_{\partial_0\partial}(t)")

        ax[:xlim]([xmin,xmax]); ax[:ylim]([-1.1;1.1]);

        # ----------- PLOT 6 ----------------------
        ax = axs[6]
        ax[:title]('Wolfe terms')
        ax[:plot](ts,ma,'-',c=mpg);
        ax[:plot](ts,mb,'-',c=dre);
        ax[:plot](ts,ma+2*sqrt.(Vaa),'-', c=lightmpg);
        ax[:plot](ts,ma-2*sqrt.(Vaa),'-', c=lightmpg);
        ax[:plot](ts,mb+2*sqrt.(Vbb),'-', c=lightdre);
        ax[:plot](ts,mb-2*sqrt.(Vbb),'-', c=lightdre);
        ax[:plot](ts,sqrt.(abs.(Vab)),':', c=blu);

        ax[:xlim]([xmin,xmax]);
    end
end
