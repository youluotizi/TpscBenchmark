"""
Holding struct for k-mesh and sparsely sampled imaginary time 'tau' / Matsubara frequency 'iw_n' grids.
Additionally we defines the Fourier transform routines 'r <-> k'  and 'tau <-> l <-> wn'.
"""
struct Mesh
    beta ::Float64
    t1   ::Float64
    U    ::Float64
    n    ::Float64

    nk1  ::Int64
    nk2  ::Int64
    nk   ::Int64
    
    iw0_f::Int64
    iw0_b::Int64
    fnw  ::Int64
    fntau::Int64
    bnw  ::Int64
    bntau::Int64
end

""" Initiarize mesh function """
function Mesh(
    t1 ::Float64,
    U  ::Float64,
    n  ::Float64,
    nk1::Int64,
    nk2::Int64,
    IR ::FiniteTempBasisSet;
    show_wmax::Bool=false
)
    nk = nk1*nk2

    # lowest Matsubara frequency index
    iw0_f = findall(x->x==FermionicFreq(1), IR.wn_f)[1]
    iw0_b = findall(x->x==BosonicFreq(0), IR.wn_b)[1]

    # the number of sampling point for fermion and boson
    fnw   = length(IR.smpl_wn_f.sampling_points)
    fntau = length(IR.smpl_tau_f.sampling_points)
    bnw   = length(IR.smpl_wn_b.sampling_points)
    bntau = length(IR.smpl_tau_b.sampling_points)

    if show_wmax
        print("max smpl fermion wn: ")
        display(IR.wn_f[end])
        print("max smpl bosonic wn: ")
        display(IR.wn_b[end])
    end

    beta = SparseIR.beta(IR)
    Mesh(beta,t1,U,n, nk1,nk2,nk, iw0_f,iw0_b,fnw,fntau,bnw,bntau)
end


"""
Solver struct to calculate the TPSC loop self-consistently.
After initializing the Solver by `solver = TPSCSolver(mesh,ek,IR, mu,Usp,Uch,sigma_init)`
it can be run by `solve(solver)`.
"""
mutable struct TPSCSolver
    mesh ::Mesh
    IR   ::FiniteTempBasisSet
    ek   ::Array{Float64,2}

    mu   ::Float64
    Usp  ::Float64
    Uch  ::Float64

    Gkw  ::Array{ComplexF64,3} # G(k,iωₙ)
    Grt  ::Array{ComplexF64,3} # G(r,τ)
    Xkw  ::Array{ComplexF64,3} # χ⁽⁰⁾(k,iωₙ)
    sigma::Array{ComplexF64,3} # Σ(k,iωₙ)
end

""" Initiarize function """
function TPSCSolver(
    mesh::Mesh,
    IR::FiniteTempBasisSet,
    ek::Array{Float64,2},
    sigma_init::Array{ComplexF64,3}
)
    Gkw = Array{ComplexF64}(undef, mesh.fnw,   mesh.nk1, mesh.nk2)
    Grt = Array{ComplexF64}(undef, mesh.fntau, mesh.nk1, mesh.nk2)
    Xkw = Array{ComplexF64}(undef, mesh.bnw,   mesh.nk1, mesh.nk2)

    solver = TPSCSolver(mesh,IR,ek, 0.0,0.0,0.0, Gkw,Grt,Xkw,sigma_init)

    solver.mu = cal_mu!(solver)
    cal_Gkw!(solver, solver.mu)
    cal_Grt!(solver)
    cal_Xkw!(solver)
    return solver
end

function cal_ek(t1, nk1, nk2)
    ek = Array{Float64,2}(undef, nk1, nk2)
    Threads.@threads for iy in 1:nk2
        ky = (2pi*(iy-1))/nk2
        @inbounds for ix in 1:nk1
            kx = (2pi*(ix-1))/nk1
            ek[ix,iy] = -2.0*(cos(kx)+cos(ky)+t1*cos(kx+ky))
        end
    end
    return ek
end

function _cal_Gkw!(
    Gkw::Array{ComplexF64,3},
    ek::Array{Float64,2},
    sigma::Array{ComplexF64,3},
    w::Array{ComplexF64,1}
)
    nw,nk1,nk2=size(sigma)
    Threads.@threads for iy in 1:nk2
        @inbounds for ix in 1:nk1, iw in 1:nw
            Gkw[iw,ix,iy] = 1.0/(w[iw]-ek[ix,iy]-sigma[iw,ix,iy])
        end
    end
    nothing
end
""" calculate Green function G(iw,k) """
function cal_Gkw!(solver,mu::Float64)
    β = solver.mesh.beta
    w = map(wn->valueim(wn,β)+mu, solver.IR.wn_f)
    _cal_Gkw!(solver.Gkw, solver.ek, solver.sigma, w)
    nothing
end

""" Calculate real space Green function G(τ,r) [for calculating χ₀ and Σ] """
function cal_Grt!(solver::TPSCSolver)
    Grw = k_to_r(solver.Gkw)
    solver.Grt .= wn_to_tau(solver.IR, Fermionic(), Grw)
    nothing
end

""" Calculate irreducible susciptibility chi0(iv,q) """
function cal_Xrw!(Xrt, Grt)
    nt,nk1,nk2=size(Xrt)
    Threads.@threads for iy in 1:nk2
        @inbounds for ix in 1:nk1, it in 1:nt
            Xrt[it,ix,iy] = 2*Grt[it,ix,iy]*Grt[nt-it+1,ix,iy]
        end
    end
    nothing
end
function cal_Xkw!(solver::TPSCSolver)
    Xrt = similar(solver.Grt)
    cal_Xrw!(Xrt, solver.Grt)
    Xrt .= r_to_k(Xrt) 
    solver.Xkw .= tau_to_wn(solver.IR, Bosonic(), Xrt)
    nothing
end

# --------------- Setting chemical potential mu --------------
""" Calculate electron density from Green function """
function cal_electron_density!(solver::TPSCSolver,mu::Float64)::Float64
    cal_Gkw!(solver,mu)
    Gw = dropdims(sum(solver.Gkw, dims=(2,3)),dims=(2,3)).*(1/solver.mesh.nk)

    g_l = fit(solver.IR.smpl_wn_f, Gw, dim=1)
    g_tau0 = dot(solver.IR.basis_f.u(0), g_l)

    return 2*(1.0+real(g_tau0))
end

""" Find chemical potential for a given filling n0 via brent's root finding algorithm """
function cal_mu!(solver::TPSCSolver)::Float64
    n = solver.mesh.n
    f(u) = cal_electron_density!(solver,u) - n
    mu_rng = extrema(solver.ek).*6
    mu = find_zero(f, mu_rng, Roots.Brent())
    return mu
end

""" RPA susciptibility """
function cal_RPA_term!(X_RPA::AbstractArray{ComplexF64,3},X0::AbstractArray{ComplexF64,3},U::Float64)
    Threads.@threads for ii in eachindex(X0)
        @inbounds X_RPA[ii]=X0[ii]/(1-0.5*U*X0[ii])
    end
    nothing
end
function cal_RPA_term(X0::AbstractArray{ComplexF64,3},U::Float64)
    X_RPA=similar(X0)
    cal_RPA_term!(X_RPA, X0, U)
    return X_RPA
end
function chi_trace!(
    X_RAP::AbstractArray{ComplexF64,3},
    X0::AbstractArray{ComplexF64,3},
    nk::Int,
    IR::FiniteTempBasisSet,
    U ::Float64    
)
    cal_RPA_term!(X_RAP, X0, U)
    Xio = dropdims(sum(X_RAP,dims=(2,3)),dims=(2,3)).*(1/nk)
    g_l = fit(IR.smpl_wn_b, Xio, dim=1)
    g_tau0::ComplexF64 = dot(IR.basis_b.u(0), g_l)
    return g_tau0
end


function cal_double_occ(Usp::Float64,U::Float64,n::Float64)
    if n<1.0+1e-8
        return Usp*(0.5n)^2/U
    else
        return Usp*(1-0.5n)^2/U+n-1
    end
end

function cal_spin_vertex!(solver::TPSCSolver)
    X0 = solver.Xkw
    n  = solver.mesh.n<1.0+1e-10 ? solver.mesh.n : 2.0-solver.mesh.n
    U  = solver.mesh.U
    nk = solver.mesh.nk
    IR = solver.IR

    U_crit = 2.0/maximum(real.(X0))-1e-11
    X_RPA = similar(X0) # temp variable, for efficiency
    f(x::Float64) = real(chi_trace!(X_RPA,X0,nk,IR,x))+x*n^2/(2*U)-n
    solver.Usp = find_zero(f, (0.0, U_crit), Roots.Brent())

    # f(x::Float64) = abs2(real(chi_trace!(X_RPA,X0,nk,IR,x))+x*n^2/(2*U)-n)
    # res=optimize(f, 0.0, U_crit; abs_tol=1e-12)
    # solver.Usp=Optim.minimizer(res)
end

function cal_charge_vertex!(solver::TPSCSolver)
    X0 = solver.Xkw
    n  = solver.mesh.n # <10.0+1e-6 ? solver.n : 2.0-solver.n
    U  = solver.mesh.U
    C0 = n^2-n-2*cal_double_occ(solver.Usp,U,n) #(n-1-0.5*n*solver.Usp/solver.U)*n
    nk = solver.mesh.nk
    IR = solver.IR

    X_RPA = similar(X0) # temp variable, for efficiency
    f(x::Float64) = real(chi_trace!(X_RPA,X0,nk,IR,-1*x))+C0
    solver.Uch = find_zero(f, (1e-8, 50000*U), Roots.Brent())

    # f(x::Float64) = abs2(real(chi_trace!(X_RPA,X0,nk,IR,-1*x))+C0)
    # res=optimize(f, 1e-8,10*solver.U; abs_tol=1e-12)
    # solver.Uch=Optim.minimizer(res)
end

""" calculate -x by period boundary condition """
minusx(x::Int,N::Int)= x==1 ? 1 : N+2-x

""" calculate self-energy Σ(r,τ) """
function cal_Sr(Vrtau, Grt)
    nt,nk1,nk2=size(Vrtau)
    Σr=similar(Vrtau)
    Threads.@threads for iy in 1:nk2
        iy1=minusx(iy,nk2)
        for ix in 1:nk1
            ix1=minusx(ix,nk1)
            for it in 1:nt
                Σr[it,ix,iy]=Vrtau[nt+1-it,ix1,iy1]*Grt[it,ix,iy]
            end
        end
    end
    return Σr
end

""" calculate self-energy Σ(k,iωₙ) """
function cal_sigma!(solver::TPSCSolver)
    Xsp=similar(solver.Xkw)
    Xch=similar(solver.Xkw)
    cal_RPA_term!(Xsp, solver.Xkw, solver.Usp)
    
    cal_charge_vertex!(solver)
    cal_RPA_term!(Xch, solver.Xkw, -1*solver.Uch)
    Vkw = ((3*solver.Usp) .* Xsp .+ solver.Uch .* Xch).*(solver.mesh.U/8)

    Vkw .= k_to_r(Vkw)
    Vrtau = wn_to_tau(solver.IR, Bosonic(), Vkw)

    Σr = cal_Sr(Vrtau, solver.Grt)
    Σr.= r_to_k(Σr)

    solver.sigma .= tau_to_wn(solver.IR, Fermionic(), Σr)
    solver.sigma .+= 0.5*solver.mesh.n*solver.mesh.U
    nothing
end

function smpl_obj2(IR::FiniteTempBasisSet, statistics::Statistics)
    if statistics == Fermionic()
        smpl_u = IR.basis_f.u
        smpl_wn  = IR.smpl_wn_f
    elseif statistics == Bosonic()
        smpl_u = IR.basis_b.u
        smpl_wn  = IR.smpl_wn_b
    end
    return smpl_u, smpl_wn
end
function wn_to_tau0(IR::FiniteTempBasisSet,statistics::Statistics,obj::Array{ComplexF64,3})
    smpl_u, smpl_wn = smpl_obj2(IR, statistics)
    Ul0 = smpl_u(0)
    gl = fit(smpl_wn, obj, dim=1)

    _,nk1,nk2=size(obj)
    tau0 = Array{ComplexF64}(undef, nk1, nk2)
    Threads.@threads for iy in 1:nk2
        for ix in 1:nk1
            tau0[ix,iy]=dot(Ul0,view(gl,:,ix,iy))
        end
    end
    return tau0
end
function Gkw_trace(
    Gkw::Array{ComplexF64,3},
    solver::TPSCSolver,
    statistics::Statistics
)
    smpl_u, smpl_wn = smpl_obj2(solver.IR, statistics)
    Gw = dropdims(sum(Gkw,dims=(2,3)),dims=(2,3))./solver.mesh.nk
    g_l = fit(smpl_wn, Gw, dim=1)

    β = solver.mesh.beta-1e-12
    g_tau0 = dot(smpl_u(β), g_l)*(-1)
    return g_tau0
end