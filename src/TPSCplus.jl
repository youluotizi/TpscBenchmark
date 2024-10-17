export solve_p!,solve_p2!

""" TPSC+ solution """
function solve_p!(solver::TPSCSolver,U::AbstractArray{Float64,1})
    cal_spin_vertex!(solver)
    cal_charge_vertex!(solver)
    
    U[1:2].=solver.Usp, solver.Uch
    # g0kw = copy(solver.Gkw)
    g0rt = copy(solver.Grt)
    g2kw = copy(solver.Gkw)
    tmp=0.0

    println("---------- TPSC+ -----------")
    for ii in 1:25
        cal_sigma!(solver)
        solver.mu = cal_mu!(solver)
        cal_Gkw!(solver, solver.mu)
        # solver.Gkw .= 0.5 .* g2kw .+ 0.5 .* solver.Gkw
        tmp=norm(solver.Gkw.-g2kw)
        println(ii,": ",tmp)

        tmp<1e-5 && break
        g2kw .= solver.Gkw
        cal_Grt!(solver)

        solver.Xkw.=X2_calc(solver.mesh,solver.Grt,g0rt)
        cal_spin_vertex!(solver)
        cal_charge_vertex!(solver)  
    end

    # println("Tr[ΣG]: ",gkio_trace(solver.sigma.*g2kw,solver,Fermionic()))
    # println("U⟨n↑n↓⟩:",solver.Usp*solver.n^2*0.25)

    U[3:5].=solver.Usp,solver.Uch,tmp
    Xsp = cal_RPA_term(solver.Xkw, solver.Usp) # in (iω,k) space
    Xsp.= k_to_r(Xsp)  # in (iω,r) space
    Xsp_t0 = wn_to_tau0(solver.IR, Bosonic(),Xsp)
    return pmat(Xsp_t0,3,3)
end

""" partially-dressed susciptibility χ⁽²⁾ """
function X2_calc(IR,g2,g0)
    Xr=similar(g2)
    nt,nk1,nk2=size(Xr)
    for iy in 1:nk2, ix in 1:nk1
        ix1=minusx(ix,nk1)
        iy1=minusx(iy,nk2)
        for it in 1:nt
            # Xr[it,ix,iy]=g2[nt+1-it,ix1,iy1]*g0[it,ix,iy]+g2[it,ix,iy]*g0[nt+1-it,ix1,iy1]
            Xr[it,ix,iy]=2*g2[nt+1-it,ix1,iy1]*g2[it,ix,iy]
        end
    end
    Xr.=r_to_k(Xr)
    tau_to_wn(IR, Bosonic(), Xr)
end

# TPSC+ => ⟨nn⟩ then TPSC
function solve_p2!(solver::TPSCSolver,U::AbstractArray{Float64,1})
    cal_spin_vertex!(solver)
    cal_charge_vertex!(solver)
    
    U[1:2].=solver.Usp, solver.Uch
    # g0kw = copy(solver.Gkw)
    g0rt = copy(solver.Grt)
    g2kw = copy(solver.Gkw)
    X0 = copy(solver.Xkw)
    tmp = 0.0

    println("---------- TPSC+2 -----------")
    for ii in 1:25
        cal_sigma!(solver)
        solver.mu = cal_mu!(solver)
        cal_Gkw!(solver, solver.mu)
        # solver.Gkw .= 0.5 .* g2kw .+ 0.5 .* solver.Gkw
        tmp=norm(solver.Gkw.-g2kw)
        println(ii,": ",tmp)

        tmp<1e-5 && break
        g2kw .= solver.Gkw
        cal_Grt!(solver)

        solver.Xkw.=X2_calc(solver.mesh,solver.Grt,g0rt)
        cal_spin_vertex!(solver)
        cal_charge_vertex!(solver)  
    end

    # println("Tr[ΣG]: ",gkio_trace(solver.sigma.*g2kw,solver,Fermionic()))
    # println("U⟨n↑n↓⟩:",solver.Usp*solver.n^2*0.25)

    D=solver.Usp*solver.mesh.n^2*0.25/solver.mesh.U
    U[3:4].=solver.Usp, solver.Uch
    U[5]=cal_spin_vertex(X0, solver.mesh.nk+0.0, solver.IR, solver.mesh.n, D)

    Xsp = cal_RPA_term(X0, U[5]) # in (iω,k) space
    Xsp.= k_to_r(Xsp)  # in (iω,r) space
    Xsp_t0 = wn_to_tau0(solver.IR, Bosonic(),Xsp)
    return pmat(Xsp_t0,3,3)
end

function spin_vertex_calc2(X0,nk,IR,n,D)
    C0=n-2*D
    U_crit = 2.0/maximum(real.(X0))-1e-11
    X_RPA = similar(X0) # temp variable, for efficiency
    f(x::Float64) = real(chi_trace!(X_RPA,X0,nk,IR,x))-C0
    find_zero(f, (0.0, U_crit), Roots.Brent())
end