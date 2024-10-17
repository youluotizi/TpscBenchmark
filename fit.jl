using LinearAlgebra, MKL
using OrderedCollections, FileIO
using Revise
using CSV,DataFrames
using TOML
using Optim
using BlackBoxOptim

includet("src/TPSC.jl")
using .TPSC
include("src/Common.jl")
using .Common: LinBz,bz2d

using CairoMakie
using DelimitedFiles
using Printf
using LaTeXStrings


c = Makie.wong_colors()
skipmiss = collect∘skipmissing
skmissM(m)=hcat([skipmiss(i) for i in eachcol(m)]...)
D_to_M(m)=hcat(eachcol(m)...)
function mysort!(x::AbstractVector{<:Number},y...)
    pt=sortperm(x)
    x.=x[pt]
    for ii in y
        _mysort!(pt,ii)
    end
end
_mysort!(pt::Vector{Int},V::Vector{<:Number})=(V.=V[pt])
_mysort!(pt::Vector{Int},V::Matrix{<:Number})=(V.=V[:,pt])
_mysort!(pt::Vector{Int},V::Array{<:Number,3})=(V.=V[:,:,pt])


##  exp data plot
t1,T,U = 0.97, 0.35, 9.0
pname = @sprintf("Dope_tp%.2f_U%.2f_T%.2f", t1,U,T)
plist=Dict("Dope_tp0.03_U9.00_T0.35"=>[3,22,5,8, 3],
    "Dope_tp0.57_U9.00_T0.35"=>[3,17,29,32, 3],
    "Dope_tp0.97_U9.00_T0.35"=>[3,16,14,17, 3],
    "Dope_tp0.03_U9.00_T0.40"=>[3,20,6,9, 4],
    "Dope_tp0.57_U9.00_T0.40"=>[3,22,51,54, 4],
    "Dope_tp0.97_U9.00_T0.40"=>[3,14,46,49, 4]
)
p = plist[pname]
s = CSV.read("../Source Data/fig$(p[5]).csv",DataFrame,header=[1,2])

dexp = skmissM(s[:,p[3]:p[4]])

set_theme!(size=(400,300))
begin
    fig,ax,plt=scatter(dexp[:,1],dexp[:,2],marker=:xcross,color=:black)
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    fig
end



## dqmc plot
iplot = 1
dpath="C:/Users/zylt/Desktop/frustrated_hubbard/"*pname*"_L8"
dlist=readdir(dpath)
dn = Array{Float64}(undef,length(dlist))
sz = Array{Float64}(undef,6,length(dn))
docc = similar(dn)
docc_err = similar(dn)
for (ii,id) in enumerate(dlist)
    dqmc=CSV.read(joinpath(dpath,id,"global_stats.csv"),DataFrame)
    dn[ii]=dqmc[dqmc[!,1].=="density","MEAN_R"][1]-1
    docc[ii]=dqmc[dqmc[!,1].=="double_occ","MEAN_R"][1]
    docc_err[ii]=dqmc[dqmc[!,1].=="double_occ","STD"][1]
    sz[:,ii].=reshape(readdlm(joinpath(dpath,id,"equal-time/spin_z/spin_z_position_equal-time_stats.csv"))[[3,11,19],[6,8]],:)
end
mysort!(dn,docc,sz)

begin
    scatterlines!(dn,sz[iplot,:],color=c[iplot],linestyle=:dash,markersize=7)
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[iplot])
    fig
end


## -------------------   fit Cr_n plot   --------------------------
#= general fitting
function main(t1, T, U, dn,sz,r)
    n=dn.+1
    Ndata = length(n)
    Ulist = Array{Float64}(undef, 2,Ndata)
    Cr = Array{ComplexF64}(undef, 6, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.4     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)
    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))
    X_RPA = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2))
    X_t0= Array{ComplexF64}(undef,(nk1,nk2))
    for ii in 1:Ndata
        println(ii,", ",n[ii])
        msh = TPSC.Mesh(t1,U,n[ii], nk1,nk2, IR)
        
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)
        U_crit = 2.0/maximum(real.(solver.Xkw))-1e-10

        f(Usp)=spin_vertex_fit!(X_RPA,X_t0,solver.Xkw,Usp,sz[:,ii],r,IR)
        res=optimize(f, 0.0, U_crit; abs_tol=1e-12)
        Ulist[1,ii]=Optim.minimizer(res)
        
        # res=bboptimize(f; SearchRange=(0.0,U_crit),NumDimensions=1,MaxFuncEvals=2000,TraceMode=:silent)
        # Ulist[ii]=best_candidate(res)[1]
        Cr[1:3,ii].=X_t0[2,1],X_t0[2,2],X_t0[3,2]

        Cd=solve!(solver, Xcut=3)
        Ulist[2,ii] = solver.Usp
        Cr[4:6,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
    end
    return (;Ulist, Cr)
end
=#
# fitting by quadratic function
function Gamma_to_Cr(t1, T, U, dn,ΔΓ,n0)
    n=dn.+1
    Ndata = length(n)
    Γtpsc = Array{Float64}(undef, Ndata)
    sz_tpsc = Array{ComplexF64}(undef, 3, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.4     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)

    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))
    X0 = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2, Ndata))
    docc = Array{Float64}(undef,Ndata)
    docc_fit = similar(docc)

    X_RPA = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2))
    Γ_fit = similar(Γtpsc)
    sz_fit = similar(sz_tpsc)

    for ii in 1:Ndata
        msh = TPSC.Mesh(t1,U,n[ii], nk1,nk2, IR)
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)

        Cd=solve!(solver, Xcut=3)
        Γtpsc[ii] = solver.Usp
        sz_tpsc[:,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        docc[ii] = cal_docc!(X_RPA,solver.Xkw,nk1*nk2,IR,Γtpsc[ii],n[ii])

        Γ_fit[ii]=TPSC.fn(Γtpsc[ii],n[ii],ΔΓ,n0)
        docc_fit[ii] = cal_docc!(X_RPA,solver.Xkw,nk1*nk2,IR,Γ_fit[ii],n[ii])
        TPSC.cal_RPA_term!(X_RPA,solver.Xkw, Γ_fit[ii]) # in (iω,k) space
        X_RPA.= TPSC.k_to_r(X_RPA)  # in (iω,r) space
        Xsp_t0 = TPSC.wn_to_tau0(solver.IR, TPSC.Bosonic(),X_RPA)
        sz_fit[:,ii] .= Xsp_t0[2,1],Xsp_t0[2,2],Xsp_t0[3,2]
    end

    return (;Γtpsc, Γ_fit, sz_tpsc, sz_fit, docc,docc_fit)
end

function main(t1, T, U, dn,sz_dqmc,r,n0)
    n=dn.+1
    Ndata = length(n)
    Γtpsc = Array{Float64}(undef, Ndata)
    sz_tpsc = Array{ComplexF64}(undef, 3, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.4     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)

    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))
    X0 = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2, Ndata))
    docc = Array{Float64}(undef,Ndata)
    docc_fit = similar(docc)

    for ii in 1:Ndata
        msh = TPSC.Mesh(t1,U,n[ii], nk1,nk2, IR)
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)

        Cd=solve!(solver, Xcut=3)
        Γtpsc[ii] = solver.Usp
        sz_tpsc[:,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        X0[:,:,:,ii].=solver.Xkw
    end

    X_RPA = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2))
    Γ_fit = similar(Γtpsc)
    sz_fit = similar(sz_tpsc)
    f(ΔΓ)=spin_vertex_fit_line!(ΔΓ, X_RPA,X0, Γ_fit,Γtpsc, n,n0, sz_dqmc,sz_fit,r,IR)
    res=optimize(f, 0.0, 10.0; abs_tol=1e-12)
    dΓ = Optim.minimizer(res)

    for ii in 1:Ndata
        docc[ii] = cal_docc!(X_RPA,view(X0,:,:,:,ii),nk1*nk2,IR,Γtpsc[ii],n[ii])
        docc_fit[ii] = cal_docc!(X_RPA,view(X0,:,:,:,ii),nk1*nk2,IR,Γ_fit[ii],n[ii])
    end

    return (;dΓ, Γtpsc, Γ_fit, sz_tpsc, sz_fit, docc,docc_fit)
end

r=[1.0,0.0,0.0]
a3 = main(t1, T, U, dn, sz,r,0.0)

dn2 = range(-0.95,0.95,39)
af = Gamma_to_Cr(t1, T, U, dn2,a3.dΓ,0.0)

iplot = 1

# original fit data plot
begin
    fig = Figure(size=(500,400))
    gamma = @sprintf "%.2f\$" a3.dΓ
    ax = Axis(fig[1,1],title=latexstring("\$\\delta_0=0\$, \$a_\\text{fit}="*gamma),titlesize=20)
    scatter!(dexp[:,1],dexp[:,2],marker=:xcross,color=:black,label="exp")
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    scatterlines!(dn,sz[iplot,:],color=c[1],linestyle=:dash,markersize=7,label="dqmc")
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[1])
    scatterlines!(dn,real.(a3.sz_tpsc[iplot,:]),markersize=7,linestyle=:dot,color=c[2],label="tpsc")
    scatterlines!(dn,real.(a3.sz_fit[iplot,:]),markersize=7,color=c[3],label="fit")
    axislegend(position=:rb)
    fig
end
# more delta plot for tpsc and fitting 
begin
    fig = Figure(size=(500,400))
    gamma = @sprintf "%.2f\$" a3.dΓ
    ax = Axis(fig[1,1],title=latexstring("\$\\delta_0=0\$, \$a_\\text{fit}="*gamma),titlesize=20)
    scatter!(dexp[:,1],dexp[:,2],marker=:xcross,color=:black,label="exp")
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    scatterlines!(dn,sz[iplot,:],color=c[1],linestyle=:dash,markersize=7,label="dqmc")
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[1])
    scatterlines!(dn2,real.(af.sz_tpsc[iplot,:]),markersize=7,linestyle=:dot,color=c[2],label="tpsc")
    scatterlines!(dn2,real.(af.sz_fit[iplot,:]),markersize=7,color=c[3],label="fit")
    axislegend(position=:rb)
    fig
end


begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1],title=latexstring("vertex \$\\Gamma\$"),titlesize=20)
    scatterlines!(dn2,af.Γtpsc,color=c[1],markersize=7,linestyle=:dash,label="tpsc")
    scatterlines!(dn2,af.Γ_fit,markersize=7,color=c[2],
    label=latexstring("\$\\delta_0=0\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a3.dΓ)))
    # scatterlines!(dn2,a2.Γ_fit,markersize=7,color=c[3],linestyle=:dot,label=latexstring("\$\\delta_0=-0.1\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a2.dΓ)))
    # scatterlines!(dn2,a3.Γ_fit,markersize=7,color=c[4],linestyle=:dashdot,label=latexstring("\$\\delta_0=-0.2\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a3.dΓ)))
    axislegend(position=:lb)
    fig
end

begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1],title=latexstring("double occupancy \$\\langle n_{\\uparrow}n_{\\downarrow}\\rangle\$"),titlesize=20)
    scatterlines!(dn,docc,color=:black,markersize=7,label="dqmc")
    errorbars!(dn,docc,docc_err,whiskerwidth = 10,color=:black)
    scatterlines!(dn2,af.docc,color=c[1],markersize=7,label="tpsc",linestyle=:dash)
    scatterlines!(dn2,af.docc_fit,markersize=7,color=c[2],label=latexstring("\$\\delta_0=0\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a3.dΓ)))
    # scatterlines!(dn,a2.docc_fit,markersize=7,color=c[3],linestyle=:dot,label=latexstring("\$\\delta_0=-0.1\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a2.dΓ)))
    # scatterlines!(dn,a3.docc_fit,markersize=7,color=c[4],linestyle=:dashdot,label=latexstring("\$\\delta_0=-0.2\$, \$a_\\text{fit}="*@sprintf("%.2f\$",a3.dΓ)))
    axislegend(position=:lt)
    fig
end

save("data/Fit.h5",OrderedDict(
    "expt"=>dexp,
    "dn"=>dn,
    "dsz"=>sz,
    "d_docc"=>docc,
    "d_docc_err"=>docc_err,
    "n"=>collect(dn2),
    "sz"=>real.(af.sz_tpsc),
    "sz_fit"=>real.(af.sz_fit),
    "docc"=>af.docc,
    "docc_fit"=>af.docc_fit,
    "Utpsc"=>af.Γtpsc,
    "Ufit"=>af.Γ_fit,
    "n0"=>0.0,
    "DeltaGamma"=>a3.dΓ)
)



##
# fitting by quartic function
function Gamma_to_Cr(t1, T, U, dn,xsol)
    n=dn.+1
    Ndata = length(n)
    Γtpsc = Array{Float64}(undef, Ndata)
    sz_tpsc = Array{ComplexF64}(undef, 3, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.4     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)

    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))
    X0 = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2, Ndata))
    docc = Array{Float64}(undef,Ndata)
    docc_fit = similar(docc)

    X_RPA = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2))
    Γ_fit = similar(Γtpsc)
    sz_fit = similar(sz_tpsc)

    for ii in 1:Ndata
        msh = TPSC.Mesh(t1,U,n[ii], nk1,nk2, IR)
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)

        Cd=solve!(solver, Xcut=3)
        Γtpsc[ii] = solver.Usp
        sz_tpsc[:,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        docc[ii] = cal_docc!(X_RPA,solver.Xkw,nk1*nk2,IR,Γtpsc[ii],n[ii])

        Γ_fit[ii]=TPSC.fn2(Γtpsc[ii],n[ii]-1,xsol)
        docc_fit[ii] = cal_docc!(X_RPA,solver.Xkw,nk1*nk2,IR,Γ_fit[ii],n[ii])
        TPSC.cal_RPA_term!(X_RPA,solver.Xkw, Γ_fit[ii]) # in (iω,k) space
        X_RPA.= TPSC.k_to_r(X_RPA)  # in (iω,r) space
        Xsp_t0 = TPSC.wn_to_tau0(solver.IR, TPSC.Bosonic(),X_RPA)
        sz_fit[:,ii] .= Xsp_t0[2,1],Xsp_t0[2,2],Xsp_t0[3,2]
    end

    return (;Γtpsc, Γ_fit, sz_tpsc, sz_fit, docc,docc_fit)
end
##
function main(t1, T, U, dn,sz_dqmc,r)
    n=dn.+1
    Ndata = length(n)
    Γtpsc = Array{Float64}(undef, Ndata)
    sz_tpsc = Array{ComplexF64}(undef, 3, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 64,64
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.3     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)

    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))
    X0 = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2, Ndata))
    docc = Array{Float64}(undef,Ndata)
    docc_fit = similar(docc)

    for ii in 1:Ndata
        msh = TPSC.Mesh(t1,U,n[ii], nk1,nk2, IR)
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)

        Cd=solve!(solver, Xcut=3)
        Γtpsc[ii] = solver.Usp
        sz_tpsc[:,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        X0[:,:,:,ii].=solver.Xkw
    end

    X_RPA = Array{ComplexF64}(undef,(length(IR.wn_b),nk1,nk2))
    Γ_fit = similar(Γtpsc)
    sz_fit = similar(sz_tpsc)
    f(x)=TPSC.spin_vertex_fit_line2!(x, X_RPA,X0, Γ_fit,Γtpsc, n,sz_dqmc,sz_fit,r,IR)
    res=optimize(f, [1.0,-0.4,0.0])
    dΓ = Optim.minimizer(res)

    for ii in 1:Ndata
        docc[ii] = cal_docc!(X_RPA,view(X0,:,:,:,ii),nk1*nk2,IR,Γtpsc[ii],n[ii])
        docc_fit[ii] = cal_docc!(X_RPA,view(X0,:,:,:,ii),nk1*nk2,IR,Γ_fit[ii],n[ii])
    end

    return (;dΓ, Γtpsc, Γ_fit, sz_tpsc, sz_fit, docc,docc_fit)
end

r=[1.0,0.0,0.0]
a3 = main(t1, T, U, dn, sz,r);
dn2 = range(-0.95,0.95,39)
af = Gamma_to_Cr(t1, T, U, dn2,a3.dΓ);

##
iplot = 1
# original fit data plot
begin
    fig = Figure(size=(500,400))
    # gamma = @sprintf "%.2f\$" a3.dΓ
    ax = Axis(fig[1,1])
    scatter!(dexp[:,1],dexp[:,2],marker=:xcross,color=:black,label="exp")
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    scatterlines!(dn,sz[iplot,:],color=c[1],linestyle=:dash,markersize=7,label="dqmc")
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[1])
    scatterlines!(dn,real.(a3.sz_tpsc[iplot,:]),markersize=7,linestyle=:dot,color=c[2],label="tpsc")
    scatterlines!(dn,real.(a3.sz_fit[iplot,:]),markersize=7,color=c[3],label="fit")
    axislegend(position=:rb)
    fig
end
# more delta plot for tpsc and fitting 
begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1])
    scatter!(dexp[:,1],dexp[:,2],marker=:xcross,color=:black,label="exp")
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    scatterlines!(dn,sz[iplot,:],color=c[1],linestyle=:dash,markersize=7,label="dqmc")
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[1])
    scatterlines!(dn2,real.(af.sz_tpsc[iplot,:]),markersize=7,linestyle=:dot,color=c[2],label="tpsc")
    scatterlines!(dn2,real.(af.sz_fit[iplot,:]),markersize=7,color=c[3],label="fit")
    axislegend(position=:rb)
    fig
end


begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1])
    scatterlines!(dn2,af.Γtpsc,color=c[1],markersize=7,linestyle=:dash,label="tpsc")
    scatterlines!(dn2,af.Γ_fit,markersize=7,color=c[2],label="fit")
    axislegend(position=:lb)
    fig
end

begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1])#,title=latexstring("double occupancy \$\\langle n_{\\uparrow}n_{\\downarrow}\\rangle\$"),titlesize=20)
    scatterlines!(dn,docc,color=:black,markersize=7,label="dqmc")
    # errorbars!(dn,docc,docc_err,whiskerwidth = 10,color=:black)
    scatterlines!(dn2,af.docc,color=c[1],markersize=7,label="tpsc",linestyle=:dash)
    scatterlines!(dn2,af.docc_fit,markersize=7,color=c[2],label="fit")
    axislegend(position=:lt)
    fig
end


save("data/Fit2.h5",OrderedDict(
    "expt"=>dexp,
    "dn"=>dn,
    "dsz"=>sz,
    "d_docc"=>docc,
    "d_docc_err"=>docc_err,
    "n"=>collect(dn2),
    "sz"=>real.(af.sz_tpsc),
    "sz_fit"=>real.(af.sz_fit),
    "docc"=>af.docc,
    "docc_fit"=>af.docc_fit,
    "Utpsc"=>af.Γtpsc,
    "Ufit"=>af.Γ_fit,
    "n0"=>0.0,
    "DeltaGamma"=>a3.dΓ)
)






## ------------- fit Cr_t1 plot -----------------------
function main2(T,U)
    n = 1.0
    path = "C:/Users/zylt/Desktop/frustrated_hubbard/";
    name = path*@sprintf("tp2_U%.2f_T%.2f_L8", U, T)
    file = "global_stats.csv"
    data = readdir(name)
    Ndata = length(data)
    t1 = Array{Float64}(undef, Ndata)
    Ulist = Array{Float64}(undef, 2, Ndata)
    Cr = Array{ComplexF64}(undef, 6, Ndata)

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128

    ek = cal_ek(1.0, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.33     # set wmax >= W
    IR = FiniteTempBasisSet(beta, wmax, IR_tol)

    for ii in 1:Ndata
        hop=TOML.parsefile(joinpath(name,data[ii],"model_summary.toml"))["TightBindingModel"]["hopping"]
        t1[ii] = hop[findfirst(x->x["HOPPING_ID"]==3,hop)]["t_mean"]
        println(ii,", ",t1[ii])

        dqmc = CSV.read(joinpath(name,data[ii],file),DataFrame)
        dn   = dqmc[dqmc[!,1].=="density","MEAN_R"][1]
        ddoc = dqmc[dqmc[!,1].=="double_occ","MEAN_R"][1]

        ek = cal_ek(t1[ii], nk1, nk2)
        msh = TPSC.Mesh(t1[ii],U,n, nk1,nk2, IR)
        sigma = zeros(ComplexF64,(msh.fnw, nk1, nk2))

        # initialize calculation
        solver = TPSCSolver(msh, IR, ek, sigma)
        Cd = solve!(solver, Xcut=3)
        Cr[1:3,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        Ulist[1,ii] = solver.Usp

        sigma .= 0.0im
        Cd .= solve_DQMC!(solver, ddoc, Xcut=3)
        Cr[4:6,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        Ulist[2,ii] = solver.Usp
    end
    mysort!(copy(t1), Ulist)
    mysort!(t1, Cr)
    return (;t1, Ulist, Cr)
end

T,U = 0.35, 9.5
a = main2(T,U)

set_theme!(size=(400,300))
c=Makie.wong_colors()


begin
    fig,ax,plt = scatterlines(a.t1,real.(a.Cr[1,:]),color=:blue,markersize=7)
    for ii in 2:3
        scatterlines!(a.t1,real.(a.Cr[ii,:]),color=:blue,markersize=7)
    end
    for ii in 4:6
        scatterlines!(a.t1,real.(a.Cr[ii,:]),color=:red,markersize=7)
    end
    fig
end

#=
begin
    fig,ax,plt = scatter(a.t1,a.Ulist[1,:],color=:blue)
    scatter!(a.t1,a.Ulist[2,:],color=:red)
    fig
end
=#

##  exp data plot
dpath="C:/Users/zylt/Desktop/update/Ferromagnetism/Source Data/fig2.csv"
s = CSV.read(dpath,DataFrame,header=[1,2])
dexp = skmissM(s[:,55:61])

for ii in 1:3
    scatter!(dexp[:,7],dexp[:,2ii-1],marker=:xcross,color=:black)
    errorbars!(dexp[:,7],dexp[:,2ii-1],dexp[:,2ii],whiskerwidth = 10,color=:black)
end
fig


## dqmc plot 
mu = U/2
path = "C:/Users/zylt/Desktop/frustrated_hubbard/"
name = path*@sprintf("tp_mu%.2f_U%.2f_T%.2f_L8", mu, U, T)
file = "global_stats.csv"
data = readdir(name)

Ndata = length(data)
t1 = Array{Float64}(undef, Ndata)
Cr = Array{Float64}(undef, 6, Ndata)
for (ii,id) in enumerate(data)
    hop=TOML.parsefile(joinpath(name,id,"model_summary.toml"))["TightBindingModel"]["hopping"]
    t1[ii] = hop[findfirst(x->x["HOPPING_ID"]==3,hop)]["t_mean"]
    Cr[:,ii].=reshape(readdlm(joinpath(name,id,"equal-time/spin_z/spin_z_position_equal-time_stats.csv"))[[3,11,19],[6,8]],:)
end
mysort!(t1,Cr)

for ii in 1:3
    scatterlines!(t1,Cr[ii,:],color=c[3],linestyle=:dash,markersize=7)
    errorbars!(t1,Cr[ii,:],Cr[ii+3,:],whiskerwidth=5,color=c[3])
end
fig







## --------------- Sq plot --------------------
function main3(t1, T, U)
    Nx = 30
    Cr = Array{ComplexF64}(undef, 2*Nx+1,2*Nx+1,2)
    Ulist=Array{Float64}(undef,2)

    path = "C:/Users/zylt/Desktop/frustrated_hubbard/fig4/"
    name = path*@sprintf("tp%.2f_U%.2f_T%.2f_L8-1", t1, U, T)
    file = "global_stats.csv"

    beta = 1/T    # inverse temperature
    IR_tol= 1e-12 # accuary for l-cutoff of IR basis functions
    nk1,nk2 = 128,128
    ek = cal_ek(t1, nk1, nk2)
    W = extrema(ek)
    wmax = (W[2]-W[1])*1.4     # set wmax >= W
    println("band_width:", W[2]-W[1],", wmax:", wmax)
    IR = FiniteTempBasisSet(beta, wmax, IR_tol)
    sigma = zeros(ComplexF64,(length(IR.wn_f),nk1,nk2))

    dqmc = CSV.read(joinpath(name,file),DataFrame)
    dn = dqmc[dqmc[!,1].=="density","MEAN_R"][1]
    println("dn=", dn)
    dn = 1.0
    ddoc = dqmc[dqmc[!,1].=="double_occ","MEAN_R"][1]
    msh = TPSC.Mesh(t1,U,dn, nk1,nk2, IR)

    # initialize calculation
    sigma .= 0.0im
    solver = TPSCSolver(msh, IR,ek, sigma)
    Cr[:,:,1]=solve!(solver, Xcut=Nx)
    Ulist[1] = solver.Usp

    sigma .= 0.0im
    Cr[:,:,2].=solve_DQMC!(solver, ddoc, Xcut=Nx)
    Ulist[2] = solver.Usp

    return (;Ulist, Cr)
end
##

p=[[0.0265,0.26,9.7],[0.0265,0.39,9.7],[0.57,0.34,8.2],[0.75,0.32,8.2],[0.97,0.39,9.2]]

klist=range(-1.25pi*√2, 1.25pi*√2, length=65)
Nx = 2*30+1
Cr = Array{ComplexF64}(undef,Nx,Nx,2,5)
Sq = Array{ComplexF64}(undef,65,65,2,5)
Ulist=Array{Float64}(undef,2,5)
for (ii,ip) in enumerate(p)
    a = main3(ip...)
    Cr[:,:,:,ii].=a.Cr
    Sq[:,:,1,ii].=TPSC.Cd_to_Sq(a.Cr[:,:,1],klist,klist)
    Sq[:,:,2,ii].=TPSC.Cd_to_Sq(a.Cr[:,:,2],klist,klist)
    Ulist[:,ii].=a.Ulist
end

##
save("data/dSq.h5", OrderedDict(
    "klist"=>collect(klist),
    "tp00265T026U97"=>real.(Sq[:,:,:,1]),
    "tp00265T039U97"=>real.(Sq[:,:,:,2]),
    "tp057T034U82"=>real.(Sq[:,:,:,3]),
    "tp075T032U82"=>real.(Sq[:,:,:,4]),
    "tp097T039U92"=>real.(Sq[:,:,:,5])
))


kl_exp=LinBz([[0.0,0.0],[cos(2pi/3),sin(2pi/3)],[cos(pi/3),sin(pi/3)],[0.0,0.0]].*sqrt(4/3)*pi,64)
save("data/kl_exp.h5", OrderedDict("kk"=>kl_exp.kk,"rr"=>kl_exp.rr,"hpt"=>kl_exp.hpt))

##
kl=LinBz([[0.0,0.0],[cos(2pi/3),sin(2pi/3)],[cos(pi/3),sin(pi/3)],[0.0,0.0]].*sqrt(4/3)*pi,256)
Sq = Array{ComplexF64}(undef,256,2,5)
for ii in 1:5
    Sq[:,1,ii].=TPSC.Cd_to_Sq(Cr[:,:,1,ii],kl.kk,dmax=20.0)
    Sq[:,2,ii].=TPSC.Cd_to_Sq(Cr[:,:,2,ii],kl.kk,dmax=20.0)
end
fig,ax,plt=scatterlines(kl.rr,real.(Sq[:,1,2]))
scatterlines!(kl.rr,real.(Sq[:,2,2]))
fig


##
save("data/dSqklist.h5", OrderedDict(
    "kk"=>kl.kk,"rr"=>kl.rr,"hpt"=>kl.hpt,
    "tp00265T026U97"=>real.(Sq[:,:,1]),
    "tp00265T039U97"=>real.(Sq[:,:,2]),
    "tp057T034U82"=>real.(Sq[:,:,3]),
    "tp075T032U82"=>real.(Sq[:,:,4]),
    "tp097T039U92"=>real.(Sq[:,:,5])
))
