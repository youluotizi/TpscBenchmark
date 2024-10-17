using LinearAlgebra, MKL
using OrderedCollections, FileIO
using Printf
using Revise

includet("src/TPSC.jl")
using .TPSC

using CairoMakie
set_theme!(; size=(440,380))
##



function main3(t1, T)
    n = 1.0          # electron filling
    U = 9.5          # Hubbard interaction
    nk1,nk2 = 128,128

    β = 1/T    # inverse temperature
    IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions

    Nt1 = length(t1)
    Cd=Array{ComplexF64}(undef, 7,7,Nt1)
    Ulist=Array{Float64}(undef, 2,Nt1)
    for ii in 1:Nt1
        println("\nii=",ii)
        ek = cal_ek(t1[ii], nk1, nk2)
        W = extrema(ek)
        wmax = (W[2]-W[1])*1.3     # set wmax >= W
        IR = FiniteTempBasisSet(β, wmax, IR_tol)

        msh = TPSC.Mesh(t1[ii],U,n, nk1, nk2, IR)
        sigma_init = zeros(ComplexF64,(length(IR.wn_f), nk1, nk2))

        solver = TPSCSolver(msh, IR, ek, sigma_init)
        Cd[:,:,ii].=solve!(solver, Xcut=3)
        Ulist[:,ii].=solver.Usp,solver.Uch
    end
    (; Ulist, Cd)
end


begin
    t1= range(0.0265,0.97,28)
    T = [0.3, 0.35]          # temperature
    Cdlist=Array{Float64}(undef, 7,7,length(t1),length(T));
    ulist=Array{Float64}(undef,2,length(t1),length(T));
    for (ii,it) in enumerate(T)
        s = main3(t1, it)
        Cdlist[:,:,:,ii].=real.(s.Cd)
        ulist[:,:,ii].=s.Ulist
    end
end;

begin
    fig=Figure()
    Axis(fig[1,1])
    colors=Makie.wong_colors()
    markers=[:circle,:rect,:xcross]
    for ii in 1:length(T)
        scatterlines!(t1,Cdlist[4,5,:,ii], color=colors[1],marker=markers[ii])
        scatterlines!(t1,Cdlist[5,5,:,ii], color=colors[2],marker=markers[ii])
        scatterlines!(t1,Cdlist[6,5,:,ii], color=colors[3],marker=markers[ii])
    end
    fig
end

begin
    fig=Figure()
    Axis(fig[1,1])
    markers=[:circle,:rect,:xcross]
    for ii in axes(ulist,3)
        scatterlines!(t1,ulist[1,:,ii], marker=markers[ii])
    end
    fig
end

##
save("data/Cd_t1.h5", OrderedDict(
    "t1"=>collect(t1),
    "T"=>T,
    "U"=>9.5,
    "Cd"=>Cdlist)
)
