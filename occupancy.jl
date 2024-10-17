using LinearAlgebra, MKL
using OrderedCollections, FileIO
using Revise
using Printf

includet("src/TPSC.jl")
using .TPSC

using CairoMakie
##


function main(t1, T, U, nlist; particle_hole::Bool=false)
    β = 1/T           # inverse temperature
    IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions
    lenN = length(nlist)
    nk1,nk2  = 128,128 

    Cd=Array{Float64}(undef,7,7,lenN)
    Ulist=Array{Float64}(undef, 2, lenN)

    for ii in 1:lenN
        println("ii=",ii)
        if particle_hole && nlist[ii]>1.0+1e-10
            n=2-nlist[ii]; t=-t1
        else
            n=nlist[ii]; t=t1
        end

        ek = cal_ek(t, nk1, nk2)
        W = extrema(ek)
        wmax = (W[2]-W[1])*1.3     # set wmax >= W
        IR = FiniteTempBasisSet(β, wmax, IR_tol)

        msh = TPSC.Mesh(t,U,n, nk1, nk2, IR)
        sigma_init = zeros(ComplexF64,(length(IR.wn_f), nk1, nk2))

        solver = TPSCSolver(msh, IR, ek, sigma_init)
        Cd[:,:,ii].=real.(solve!(solver, Xcut=3))
        Ulist[:,ii].=solver.Usp,solver.Uch
    end
    (; Ulist, Cd)
end



# --------   check particle_hole transform calculation   ----------
begin
    t1 = 0.97
    U  = 9.5
    T  = 0.35
    nlist= range(0.1,1.9,15)
    s1 = main(t1, T, U, nlist, particle_hole=false)
    s2 = main(t1, T, U, nlist, particle_hole=true)
end;

begin
    fig=Figure(size=(500,400))
    ax = Axis(fig[1,1])
    scatterlines!(nlist.-1,s1.Ulist[1,:],label="eq.(5) for particle-doping")
    scatterlines!(nlist.-1,s2.Ulist[1,:], marker=:xcross,label="eq.(4) for particle-doping")
    axislegend(position=:lb)
    fig
end


begin
    fig = Figure(size=(600,400))
    ax=Axis(fig[1, 1])
    scatterlines!(nlist.-1,s1.Cd[4,5,:],linewidth=3,
        markersize=15,label="eq.(4) for particle-doping",)
    scatter!(nlist.-1, s2.Cd[4,5,:],color=:red,marker=:xcross,markersize=14,
        label="t'→-t',n→2-n, eq.(5) for particle-doping")
    axislegend(position=:lt)
    fig 
end

##

save("data/Usp_n3.h5", OrderedDict(
    "Usp"=>ulist,
    "n"=>collect(nlist.-1),
    "readme"=>"U[n,t1,T],t1=[0,0.5,1],T=[0.4,0.35],U:9.5")
)




## ----------------   Figure 5   -----------------
begin
    t1 = [0.03,0.57,0.97]
    T  = [0.35,0.4]
    U  = 9.0
    nlist= range(0.1,1.9,15)
    Cd = Array{Float64}(undef, 7,7,length(nlist),length(t1),length(T))
    Ulist = Array{Float64}(undef,2,length(nlist),length(t1),length(T))
    for iT in axes(Cd,5), it1 in axes(Cd,4)
        s = main(t1[it1], T[iT], U, nlist)
        Cd[:,:,:,it1,iT].=s.Cd
        Ulist[:,:,it1,iT].=s.Ulist
    end
end;



for iT in axes(Cd,5), it1 in axes(Cd,4)
    scatterlines(nlist.-1, Cd[4,5,:,it1,iT],linewidth=3,
        axis=(;title="C_{1,0}, "*@sprintf("U=%.2f,t'=%.2f,T=%.2f",U,t1[it1],T[iT])),
        markersize=15,figure=(;szie=(600,400))
    )|>display
end

for iT in axes(Cd,5), it1 in axes(Cd,4)
    scatterlines(nlist.-1, Cd[5,5,:,it1,iT],linewidth=3,
        axis=(;title="C_{1,1}, "*@sprintf("U=%.2f,t'=%.2f,T=%.2f",U,t1[it1],T[iT])),
        markersize=15,figure=(;szie=(600,400))
    )|>display
end




begin
    readme = String[]
    idx = 0
    for iT in axes(Cd,5), it1 in axes(Cd,4)
        idx+=1
        push!(readme, "$idx => (t':$(t1[it1]) T:$(T[iT]) U:$U)")
    end
end

save("data/Cd_n.h5", OrderedDict(
    "readme"=>readme,
    "n"=>collect(nlist.-1),
    "Cd"=>reshape(Cd, 7,7,length(nlist),:),
    "Ulist"=>reshape(Ulist,2,length(nlist),:))
)




## ------------ C_{0,0} -----------
begin
    tmp=[0.0453407,0.0551175,0.0516602,0.0564653]
    labels=["t'=0.026","t'=0.57","t'=0.75","t'=0.97"]
    markers=[:circle,:cross,:rect,:star5]
    fig = Figure()
    ax=Axis(fig[1, 1])
    for ii in 1:4
        scatterlines!((nlist.-1),Dlist[:,2,ii],
        markersize=11,label=labels[ii],marker=markers[ii])
        scatter!(0,tmp[ii], marker = markers[ii], markersize = 12)
    end
    axislegend()
    fig
end

