# compare two spreadtest executables at variety of dim, w{prec}. Barnett 6/17/24
using Printf
using CairoMakie
using JLD2        # for load/save arrays to file
using UnPack

fnam = "/home/alex/numerics/finufft/perftest/results/7-2-24-vs-ivecflags_gcc114_5700U_nthr1"   # outfile head
# locations of pair of FINUFFT repos to compare...
repo1 = "/home/alex/numerics/finufft"
repo2 = "/home/alex/numerics/nufft/finufft-svec2"

# run spreadtestnd{f} for a list of tols at one prec
# return spread & interp times as 2-by-ntols
function run_spread(repo,dim,M,N,tols,nthr,prec)
    if prec==Float64
        exec = "$repo/perftest/spreadtestnd"
    elseif prec==Float32
        exec = "$repo/perftest/spreadtestndf"    
    else error("prec not known!")
    end
    times = zeros(2,length(tols))     # spread col 1; interp col 2
    for (i,tol) in enumerate(tols)
        nr = 3                            # repetitions
        sptruns = zeros(nr)
        intruns = zeros(nr)
        for r=1:nr
            c = Cmd(`$exec $dim $M $N $tol`,env=("OMP_NUM_THREADS" => "$nthr",))
            r==1 && println(c)     # first run show entire Cmd not just strings
            o = read(c,String)     # do the cmd (no shell launched, as SGJ likes)
            sptruns[r] = parse(Float64,split(split(o,"pts in")[2],"s")[1])  # get first timing (spread) in seconds
            intruns[r] = parse(Float64,split(split(o,"pts in")[3],"s")[1])  # get 2nd timing (interp) in seconds
        end
        times[:,i] = [minimum(sptruns), minimum(intruns)]
    end
    times
end
#ts = run_spread(repo2,1,1e7,1e6,[1e-2,1e-3,1e-5],1,Float32)  # basic test
#println(ts); stop

# plots (and to PNG) all dims and w{prec} for both directions
function plot_all(fnam,ts,wstr,dims,M,N,nthr)
    ntols = length(wstr)
    repos = stack([[1,2] for i=1:ntols])   # which repo each run was from
    for dir=1:2
        dirstr = ["spread","interp"][dir]
        fig = Figure(fontsize=10, size=(1000,500))  # plot all 3 dims
        for (i,dim) in enumerate(dims)
            thrus = 1e-6 * M ./ ts[dir,i,:,:]'  # slice for this dir, dim: interleave repo1, repo2, repo1,...
            ax = Axis(fig[1,i], title="$dirstr $(dim)d M=$M N=$N $(nthr)thr")   # fnam too long
            barplot!(ax, kron(1:ntols, [1,1]), thrus[:], dodge=repos[:], color=repos[:])
            ax.xticks=(1:ntols, wstr)
            ax.xlabel="w{prec}"; ax.ylabel=L"throughput ($10^6$ NU pt/s)"
            ax.limits=((0,ntols+1),(0,nothing))
            yadd = maximum(thrus[:])       # what height to annotate % at
            for j=1:ntols       # show % change
                text!(j+0.4, yadd, text=@sprintf("%.0f%%",100*(thrus[2,j]/thrus[1,j]-1.0)), rotation=pi/2)
            end
        end
        display(fig)
        save("$(fnam)_$(dirstr)_M$(M)_N$(N).png",fig)
    end
end

# main script...........................................................................
nthr = 1; # 1: leave cpu freq at max (4.3GHz); for 8, lower to 2.7GHz since drops to this.
# set freq lim with cpupower-gui
# check with:   watch -n 1 sort -nr /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
dims = 1:3
M=1e7; N=1e6
#tolsd = [1e-3,1e-6]; tolsf = [1e-2]   # double then single prec tol lists
tolsd = [10.0^-k for k=2:14]; tolsf = [10.0^-k for k=2:5];
compute = true #false
if compute
    ntolsd = length(tolsd); ntolsf = length(tolsf)
    ntols = ntolsd+ntolsf
    ndims = length(dims)
    ts = NaN*zeros(2,ndims,ntols,2)    # timings: 1st dim = spread,interp, 4th dim = repo#
    for (i,dim) in enumerate(dims)    # do expensive runs...
        ts[:,i,1:ntolsd,1] = run_spread(repo1,dim,M,N,tolsd,nthr,Float64)
        ts[:,i,1:ntolsd,2] = run_spread(repo2,dim,M,N,tolsd,nthr,Float64)
        ts[:,i,ntolsd+1:end,1] = run_spread(repo1,dim,M,N,tolsf,nthr,Float32)
        ts[:,i,ntolsd+1:end,2] = run_spread(repo2,dim,M,N,tolsf,nthr,Float32)
        println(ts[:,i,:,:])
    end
    #tolstr = [[@sprintf "%.0e" tol for tol=tolsd]; [@sprintf "%.0ef" tol for tol=tolsf]] 
    # strings for w (nspread) for plotting...
    wstr = [[@sprintf "%d" -log10(tol)+1 for tol=tolsd]; [@sprintf "%df" -log10(tol)+1 for tol=tolsf]] 
    jldsave("$(fnam).jld2"; fnam,ts,wstr,dims,M,N,tolsd,tolsf,nthr)    # save all
    plot_all(fnam,ts,wstr,dims,M,N,nthr)
else
    f = load("$(fnam).jld2");    # gives a dict
    @unpack fnam,ts,wstr,dims,M,N,nthr = f       # not very easy way to get dict into globals
    plot_all(fnam,ts,wstr,dims,M,N,nthr)
end
