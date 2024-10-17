using LinearAlgebra #, MKL
using OrderedCollections, FileIO
using Revise
using TPSC

##
include("src/Common.jl")
using .Common: LinBz,bz2d

using CairoMakie
set_theme!(; size=(440,380))
includet("myplots.jl")

## System parameters
t1 = 0.0
T  = 0.35
n  = 1.0    # electron filling
U  = 9.0    # Hubbard interaction
nk1,nk2 = 128,128 # number of k_points along one repiprocal crystal lattice direction

β = 1/T    # inverse temperature
IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions
ek = cal_ek(t1, nk1, nk2)

W = extrema(ek)
wmax = (W[2]-W[1])*1.3     # set wmax >= W
println("band_width:", W[2]-W[1],", wmax:", wmax)
IR = FiniteTempBasisSet(β, wmax, IR_tol)


## initialize calculation
msh = TPSC.Mesh(t1,U,n, nk1,nk2, IR)
sigma_init = zeros(ComplexF64,(msh.fnw, nk1, nk2))

solver = TPSCSolver(msh, IR, ek, sigma_init);
Cd=solve!(solver,Xcut=3)

myhmap(real.(Cd))

##

klist=range(-1.25pi*√2, 1.25pi*√2, length=65)
Sq4=TPSC.Cd_to_Sq(Cd,klist,klist)
myhmap(real.(Sq4),colormap=:inferno)



##
save("data/Sq.h5", OrderedDict(
    "klist"=>collect(klist),
    "tp00265T039U97"=>real.(Sq1),
    "tp057T034U82"=>real.(Sq2),
    "tp075T032U82"=>real.(Sq3),
    "tp097T039U92"=>real.(Sq4)
))

kl_exp=LinBz([[0.0,0.0],[cos(2pi/3),sin(2pi/3)],[cos(pi/3),sin(pi/3)],[0.0,0.0]].*sqrt(4/3)*pi,64)
save("data/kl_exp.h5", OrderedDict("kk"=>kl_exp.kk,"rr"=>kl_exp.rr,"hpt"=>kl_exp.hpt))

##
kl=LinBz([[0.0,0.0],[cos(2pi/3),sin(2pi/3)],[cos(pi/3),sin(pi/3)],[0.0,0.0]].*sqrt(4/3)*pi,256)
Sq2=TPSC.Cd_to_Sq(Cd,kl.kk,dmax=20.0)
scatterlines(real.(Sq2))|>display

##
save("data/Sqklist.h5", OrderedDict(
    "kk"=>kl.kk,"rr"=>kl.rr,"hpt"=>kl.hpt,
    "tp00265T039U97"=>real.(Sq1),
    "tp097T039U92"=>real.(Sq2)
))






# ----------------------------------------
##           energy band
# ----------------------------------------

f(kx,ky,t1)=-2.0*(cos(kx)+cos(ky)+t1*cos(kx+ky))
kl=LinBz([[0.0,0.0],[1.0pi,0],[1pi,1pi],[0.0,0.0],[0.0,1.0pi],[-1pi,1pi],[0.0,0.0]],256)

function ek2d_cal(t1::Float64,n::Int)
    kmsh=bz2d([[0.0,0.0],[2pi,0.0],[0.0,2pi]],[n,n])
    ek=Array{Float64}(undef,n+1,n+1)
    for iy in axes(kmsh,3),ix in axes(kmsh,2)
        ek[ix,iy]=f(kmsh[1,ix,iy],kmsh[2,ix,iy],t1)
    end
    ek
end
ek=ek2d_cal(0.0,128);
hist(reshape(ek,:),bins=80)

ek=[f(kl.kk[1,ii],kl.kk[2,ii],1.0) for ii in axes(kl.kk,2)]
lines(kl.rr,ek,
    axis=(;xticks=(kl.rr[kl.hpt],["Γ","X₁","M₁","Γ","X₂","M₂","Γ"]),aspect=1.5)
)

function sq_to_hex(k1,k2,t)
    k1*=sqrt(2/3)
    k2*=sqrt(2)
    kx=cospi(0.25)k1-sinpi(0.25)k2
    ky=sinpi(0.25)k1+cospi(0.25)k2
    return kx,ky,t
end

kl=LinBz([[0.0,0.0],[cos(2pi/3),sin(2pi/3)],[cos(pi/3),sin(pi/3)],[0.0,0.0]].*sqrt(4/3)*pi,256)
ek=[f(sq_to_hex(kl.kk[1,ii],kl.kk[2,ii],0.0)...) for ii in axes(kl.kk,2)]
lines(kl.rr,ek,
        axis=(;xticks=(kl.rr[kl.hpt],["Γ","K₁","K₂","Γ"]),aspect=1.5)
)