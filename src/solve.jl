
""" TPSC solution """
function solve!(
    solver::TPSCSolver;
    Xcut::Int=3,
    Uch::Bool=false,
    check::Bool=false
)
    cal_spin_vertex!(solver)
    Uch && cal_charge_vertex!(solver)

    # cal_sigma!(solver)
    # solver.mu = cal_mu!(solver)
    # cal_Gkw!(solver, solver.mu)
    # cal_Grt!(solver)
    # cal_Xkw!(solver)

    Xsp = cal_RPA_term(solver.Xkw, solver.Usp) # in (iω,k) space
    Xsp.= k_to_r(Xsp)  # in (iω,r) space
    Xsp_t0 = wn_to_tau0(solver.IR, Bosonic(),Xsp)

    # check Galitski-Migdal equation
    if check
        G1=copy(solver.Gkw)
        cal_sigma!(solver)
        solver.mu = cal_mu!(solver)
        cal_Gkw!(solver, solver.mu)
        tmp=solver.sigma.*G1
        println("Tr[ΣG]: ",Gkw_trace(tmp,solver,Fermionic()))
        println("U⟨n↑n↓⟩:",solver.Usp*solver.mesh.n^2*0.25)
    end

    return pmat(Xsp_t0,Xcut,Xcut)
end



""" TPSC + DQMC """
function spin_vertex_dqmc!(solver::TPSCSolver,docc::Float64)
    X0 = solver.Xkw
    n  = solver.mesh.n
    nk = solver.mesh.nk
    IR = solver.IR

    U_crit = 2.0/maximum(real.(X0))-1e-11
    X_RPA = similar(X0) # temp variable, for efficiency
    f(x::Float64) = real(chi_trace!(X_RPA,X0,nk,IR,x))-n+2*docc
    solver.Usp = find_zero(f, (0.0, U_crit), Roots.Brent())
end
function charge_vertex_dqmc!(solver::TPSCSolver,ddoc::Float64)
    X0 = solver.Xkw
    C0 = solver.mesh.n-solver.mesh.n^2+2*ddoc
    nk = solver.mesh.nk
    IR = solver.IR

    X_RPA = similar(X0) # temp variable, for efficiency
    f(x::Float64) = real(chi_trace!(X_RPA,X0,nk,IR,-1*x))-C0
    solver.Uch = find_zero(f, (1e-8, 50000*solver.mesh.U), Roots.Brent())
end

function solve_DQMC!(
    solver::TPSCSolver,
    docc::Float64;
    Xcut::Int=3,
    U0::Float64=0.0,
    dU::Float64=1.0
)
    spin_vertex_dqmc!(solver,docc)
    # charge_vertex_dqmc!(solver,docc)

    dUsp = solver.Usp-U0
    solver.Usp = U0+dU*dUsp
    
    Xsp = cal_RPA_term(solver.Xkw, solver.Usp) # in (iω,k) space
    Xsp.= k_to_r(Xsp)  # in (iω,r) space
    Xsp_t0 = wn_to_tau0(solver.IR, Bosonic(),Xsp)

    return pmat(Xsp_t0,Xcut,Xcut)
end

function spin_vertex_test!(solver::TPSCSolver,docc::Float64)
    X0 = solver.Xkw
    n  = solver.mesh.n
    nk = solver.mesh.nk
    IR = solver.IR

    U_crit = 2.0/maximum(real.(X0))-0.1
    ulist = range(0.0, U_crit, 64)
    X_RPA = similar(X0) # temp variable, for efficiency
    flist = [chi_trace!(X_RPA,X0,nk,IR,ii) for ii in ulist]
    y = n-2*docc
    return (;y, ulist, flist)
end


function spin_vertex_fit!(X_RPA,X_t0, X0,Usp,Cr,r,IR)
    cal_RPA_term!(X_RPA, X0, Usp)

    X_RPA.= k_to_r(X_RPA)  # in (iω,r) space
    X_t0 .= wn_to_tau0(IR, Bosonic(),X_RPA)
    tmp=abs2(X_t0[2,1]-Cr[1])*r[1]
    tmp+=abs2(X_t0[2,2]-Cr[2])*r[2]
    tmp+=abs2(X_t0[3,2]-Cr[3])*r[3]
    return tmp
end

fn(Γ0::T,n::T,ΔΓ::T,n0::T) where {T<:Real} = n<n0 ? Γ0 : Γ0+ΔΓ*(n-n0)*(1-n)

# function fn2(Γ0::T,n::T,x::Vector{T}) where {T<:Real}
#     n<x[1] ? Γ0 : Γ0-(n-x[1])*(n-x[2])*(n-x[3])*(n-1)*x[4]
# end
function fn2(Γ0::T,n::T,x::Vector{T}) where {T<:Real}  
    # if n<x[2] 
    #     return Γ0
    # end
    # return Γ0-(n-x[2])^3*(n-1)*x[1]

    tmp=(n-1)*x[1]*(n+1)
    for ii in 2:length(x)
        tmp*=n-x[ii]
    end
    return Γ0-tmp
end

function spin_vertex_fit_line!(ΔΓ, X_RPA,X0, Γ_fit,Γtpsc, n,n0, sz_dqmc,sz_fit,r,IR)
    Ndata = size(sz_dqmc,2)
    tmp=0.0
    for ii in 1:Ndata
        Γ_fit[ii] = fn(Γtpsc[ii], n[ii], ΔΓ, n0-1)
        # Γ_fit[ii] = fn2(Γtpsc[ii], n[ii], ΔΓ, n0-1)
        cal_RPA_term!(X_RPA, view(X0,:,:,:,ii), Γ_fit[ii])
        X_RPA.= k_to_r(X_RPA)  # in (iω,r) space
        X_t0 = wn_to_tau0(IR, Bosonic(), X_RPA)
        sz_fit[:,ii].= X_t0[2,1],X_t0[2,2],X_t0[3,2]
        for jj in 1:3
            tmp+=abs2(sz_fit[jj,ii]-sz_dqmc[jj,ii])*r[jj]
        end
    end
    return tmp
end

function spin_vertex_fit_line2!(x, X_RPA,X0, Γ_fit,Γtpsc, n,sz_dqmc,sz_fit,r,IR)
    Ndata = size(sz_dqmc,2)
    tmp=0.0
    for ii in 1:Ndata
        # Γ_fit[ii] = fn(Γtpsc[ii], n[ii], ΔΓ, n0-1)
        Γ_fit[ii] = fn2(Γtpsc[ii], n[ii]-1, x)
        cal_RPA_term!(X_RPA, view(X0,:,:,:,ii), Γ_fit[ii])
        X_RPA.= k_to_r(X_RPA)  # in (iω,r) space
        X_t0 = wn_to_tau0(IR, Bosonic(), X_RPA)
        sz_fit[:,ii].= X_t0[2,1],X_t0[2,2],X_t0[3,2]
        for jj in 1:3
            tmp+=abs2(sz_fit[jj,ii]-sz_dqmc[jj,ii])*r[jj]
        end
    end
    return tmp
end

cal_docc!(X_RPA,X0,nk,IR,Γ,n)=0.5*(n-real(chi_trace!(X_RPA,X0,nk,IR,Γ)))

