using LinearAlgebra, MKL
using OrderedCollections, FileIO
using Revise
using CSV,DataFrames
using TOML

includet("src/TPSC.jl")
using .TPSC
include("src/Common.jl")
using .Common: LinBz,bz2d

using CairoMakie
using DelimitedFiles
using Printf

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
##


t1   = 0.03
T    = 0.35
U    = 9.0

path = "C:/Users/zylt/Desktop/frustrated_hubbard/";
name = path*@sprintf("Dope_tp%.2f_U%.2f_T%.2f_L8", t1, U, T)
file = "global_stats.csv"
data = readdir(name)
##
dqmc = CSV.read(joinpath(name,data[10],file),DataFrame)
dn   = dqmc[dqmc[!,1].=="density","MEAN_R"][1]
dmu  = dqmc[dqmc[!,1].=="chemical_potential","MEAN_R"][1]
ddoc = dqmc[dqmc[!,1].=="double_occ","MEAN_R"][1]


## System parameters
beta = 1/T    # inverse temperature
n    = dn    # electron filling
IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions
nk1,nk2 = 128,128
ek = cal_ek(t1, nk1, nk2)

W = extrema(ek)
wmax = (W[2]-W[1])*1.33     # set wmax >= W
println("band_width:", W[2]-W[1],", wmax:", wmax)
IR = FiniteTempBasisSet(beta, wmax, IR_tol)

# initialize calculation
msh = TPSC.Mesh(t1,U,n, nk1,nk2, IR);
sigma = zeros(ComplexF64,(length(IR.wn_f), nk1, nk2))
solver = TPSCSolver(msh, IR, ek, sigma);

b = TPSC.spin_vertex_test!(solver,ddoc)
scatterlines(b.ulist,real.(b.flist))
##

Ulist=zeros(5)
Cd=solve!(solver,Xcut=3)
Cd=solve_DQMC!(solver,ddoc,Xcut=3)










## ------------------ Dope sz --------------------------------
function main(t1, T, U) 
    path = "C:/Users/zylt/Desktop/frustrated_hubbard/"
    name = path*@sprintf("D2_tp%.2f_U%.2f_T%.2f_L8", t1, U, T)
    file = "global_stats.csv"
    data = readdir(name)
    Ndata = length(data)
    dn = Array{Float64}(undef, Ndata)
    Ulist = Array{Float64}(undef, 2, Ndata)
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

    for ii in 1:Ndata
        dqmc = CSV.read(joinpath(name,data[ii],file),DataFrame)
        dn[ii]= dqmc[dqmc[!,1].=="density","MEAN_R"][1]
        ddoc = dqmc[dqmc[!,1].=="double_occ","MEAN_R"][1]
        println(ii,", ",dn[ii])
        msh = TPSC.Mesh(t1,U,dn[ii], nk1,nk2, IR)

        # initialize calculation
        sigma .= 0.0im
        solver = TPSCSolver(msh, IR,ek, sigma)
        Cd=solve!(solver, Xcut=3)
        Ulist[1,ii] = solver.Usp
        Cr[1:3,ii].= Cd[4,5],Cd[5,5],Cd[6,5]

        sigma .= 0.0im
        Cd.=solve_DQMC!(solver, ddoc, Xcut=3)
        Cr[4:6,ii].= Cd[4,5],Cd[5,5],Cd[6,5]
        Ulist[2,ii] = solver.Usp
    end
    dn.-=1
    mysort!(dn,Cr)
    return (;dn, Ulist, Cr)
end

##   Dope plot,  TPSC imporve by DQMC
t1,T,U = 0.97, 0.35, 9.0
a = main(t1,T,U)

set_theme!(size=(400,300))
begin
    fig,ax,plt = scatterlines(a.dn,real.(a.Cr[2,:]),
        #,axis=(;limits=(nothing,(-0.18,0.02)))
        markersize=7
    )
    scatterlines!(a.dn,real.(a.Cr[5,:]),markersize=7)
    fig
end

#=
begin
    fig,ax,plt = scatter(a.dn,a.Ulist[1,:])
    scatter!(a.dn,a.Ulist[2,:])
    fig
end
=#


##  exp data plot
plist=Dict("Dope_tp0.03_U9.00_T0.35"=>[3,22,5,8, 3],
    "Dope_tp0.57_U9.00_T0.35"=>[3,17,29,32, 3],
    "Dope_tp0.97_U9.00_T0.35"=>[3,16,14,17, 3],
    "Dope_tp0.03_U9.00_T0.40"=>[3,20,6,9, 4],
    "Dope_tp0.57_U9.00_T0.40"=>[3,22,51,54, 4],
    "Dope_tp0.97_U9.00_T0.40"=>[3,14,46,49, 4]
)
t1,T,U = 0.03, 0.4, 9.0
pname = @sprintf("Dope_tp%.2f_U%.2f_T%.2f", t1,U,T)
p = plist[pname]
s = CSV.read("../Source Data/fig$(p[5]).csv",DataFrame,header=[1,2])

dexp = skmissM(s[:,p[3]:p[4]])

begin
    fig,_,_=scatter(dexp[:,1],dexp[:,2],marker=:xcross,color=:black)
    errorbars!(eachcol(dexp)...,whiskerwidth = 10,color=:black)
    #fig
end



## dqmc plot 
iplot = 2
dpath="C:/Users/zylt/Desktop/frustrated_hubbard/"*pname*"_L8"
dlist=readdir(dpath)

dn = Array{Float64}(undef,length(dlist))
sz = Array{Float64}(undef,6,length(dn))
for (ii,id) in enumerate(dlist)
    dqmc=CSV.read(joinpath(dpath,id,"global_stats.csv"),DataFrame)
    dn[ii]=dqmc[dqmc[!,1].=="density","MEAN_R"][1]-1
    sz[:,ii].=reshape(readdlm(joinpath(dpath,id,"equal-time/spin_z/spin_z_position_equal-time_stats.csv"))[[3,11,19],[6,8]],:)
end
mysort!(dn,sz)

begin
    scatterlines!(dn,sz[iplot,:],color=c[3],linestyle=:dash,markersize=7)
    errorbars!(dn,sz[iplot,:],sz[iplot+3,:],whiskerwidth=5,color=c[3])
    fig
end







## ------------------- t1-sz plot ----------------------

##   TPSC imporve by DQMC
function main2(T,U)
    mu= U/2
    n = 1.0

    path = "C:/Users/zylt/Desktop/frustrated_hubbard/";
    name = path*@sprintf("tp_mu%.2f_U%.2f_T%.2f_L8", mu, U, T)
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


##     exp data plot
dpath="C:/Users/zylt/Desktop/update/Ferromagnetism/Source Data/fig2.csv"
s = CSV.read(dpath,DataFrame,header=[1,2])
dexp = skmissM(s[:,55:61])

fig = Figure(size=(400,300))
ax = Axis(fig[1,1])
for ii in 1:3
    scatter!(dexp[:,7],dexp[:,2ii-1],marker=:xcross,color=c[ii])
    errorbars!(dexp[:,7],dexp[:,2ii-1],dexp[:,2ii],whiskerwidth = 10,color=c[ii])
end
fig


## dqmc plot
T,U = 0.35, 9.5
path = "C:/Users/zylt/Desktop/frustrated_hubbard/"
name = path*@sprintf("tp2_U%.2f_T%.2f_L8", U, T)
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

fig = Figure(size=(400,300))
ax = Axis(fig[1,1])
for ii in 1:1#3
    scatterlines!(t1,Cr[ii,:],color=c[ii],linestyle=:dash,markersize=7)
    errorbars!(t1,Cr[ii,:],Cr[ii+3,:],whiskerwidth=5,color=c[ii])
end
fig

##
save("data/Cd_t1_dqmc.h5", OrderedDict(
    "t1"=>t1,
    "U"=>9.5,
    "Cd_T0.30"=>Cr,
    "Cd_T0.35"=>Cr2)
)







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
