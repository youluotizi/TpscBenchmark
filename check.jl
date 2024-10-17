# ----------------------------------------
#     check the results of arXiv:cond-mat/9702188v3 
#                      and arXiv:1107.1534v2
# ----------------------------------------
using Revise
using LinearAlgebra,MKL 
using CairoMakie
set_theme!(; resolution=(500,400))

include("src/Common.jl")
using .Common: LinBz
includet("src/TPSC.jl")
using .TPSC
##


function check1()
    t1 = 1.0
    wmax = 14           # set wmax >= band weight
    T    = 0.4          # temperature
    beta = 1/T          # inverse temperature
    n    = 1.5          # electron filling
    U    = range(3.5, 4.95, 20)          # Hubbard interaction
    lenU = length(U)

    nk1, nk2  = 128, 128 
    IR_tol    = 1e-12     
    IR = FiniteTempBasisSet(beta, Float64(wmax), IR_tol)
    msh = TPSC.Mesh(t1, nk1, nk2, IR)
    sigma_init = zeros(ComplexF64,(msh.fnw, nk1, nk2))

    Ulist=Array{Float64}(undef,5,lenU)
    for ii in 1:lenU
        println("\nii=",ii)
        sigma_init.=0
        solver = TPSCSolver(msh, beta, U[ii], n, sigma_init)
        solve!(solver,view(Ulist,:,ii))
    end
    U, Ulist
end
U,ulist = check1()
series(U, ulist[1:2,:])

scatterlines(U, (ulist[2,:]),figure=(;resolution=(400,500)))

# --------------------------------------------
##            结构因子 S(q)
# --------------------------------------------
t1   = 0.0
wmax = 14     # set wmax >= W
T    = 0.2    # temperature
beta = 1/T    # inverse temperature
n    = 0.8    # electron filling
U    = 4.0    # Hubbard interaction
IR_tol= 1e-12     # accuary for l-cutoff of IR basis functions
nk1,nk2 = 128,128 # number of k_points along one repiprocal crystal lattice direction
IR = FiniteTempBasisSet(beta, Float64(wmax), IR_tol)

# initialize calculation
msh = TPSC.Mesh(t1, nk1, nk2, IR, show_wmax=false);
sigma_init = zeros(ComplexF64,(msh.fnw, nk1, nk2));
solver = TPSCSolver(msh, beta, U, n, sigma_init);

Ulist=zeros(5)
Cd=solve!(solver,Ulist,30)

kl = LinBz([[0,0.0], [1pi,0], [1pi,1pi], [0.0,0]], 64)

# 注意 TPSC.jl  函数 _Cd_to_Sq 对k空间作了个变换
Sq2=TPSC.Cd_to_Sq(Cd, kl.kk, dmax=30.0)
scatterlines(kl.rr, real.(Sq2))