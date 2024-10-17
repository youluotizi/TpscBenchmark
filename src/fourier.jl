# Check if a given function called with given types is type stable
function typestable(@nospecialize(f), @nospecialize(t))
    v = code_typed(f, t)
    stable = true
    for vi in v
        for (name, ty) in zip(vi[1].slotnames, vi[1].slottypes)
            !(ty isa Type) && continue
            if ty === Any
                stable = false
                println("Type instability is detected! the variable is $(name) ::$ty")
            end
        end
    end
    return stable
end

""" Return sampling object for given statistic """
function smpl_obj(IR::FiniteTempBasisSet, statistics::Statistics)
    if statistics == Fermionic()
        smpl_tau = IR.smpl_tau_f
        smpl_wn  = IR.smpl_wn_f
    elseif statistics == Bosonic()
        smpl_tau = IR.smpl_tau_b
        smpl_wn  = IR.smpl_wn_b
    end
    return smpl_tau, smpl_wn
end

""" Fourier transform from tau to iw_n via IR basis """
function tau_to_wn(IR::FiniteTempBasisSet, statistics::Statistics, obj_tau)
    smpl_tau,smpl_wn = smpl_obj(IR, statistics)
    obj_l = fit(smpl_tau, obj_tau, dim=1)
    obj_wn = evaluate(smpl_wn, obj_l, dim=1)
    return obj_wn
end

""" Fourier transform from iw_n to tau via IR basis """
function wn_to_tau(IR::FiniteTempBasisSet, statistics::Statistics, obj_wn)
    smpl_tau, smpl_wn = smpl_obj(IR, statistics)
    obj_l   = fit(smpl_wn, obj_wn, dim=1)
    obj_tau = evaluate(smpl_tau, obj_l, dim=1)
    return obj_tau
end

""" Fourier transform from k-space to real space, has 1/N factor """
function k_to_r(obj_k::Array{ComplexF64,3})   
    obj_r = ifft(obj_k,[2,3])
    return obj_r
end

""" Fourier transform from real space to k-space, no 1/N factor """
function r_to_k(obj_r)
    obj_k = fft(obj_r,[2,3])
    return obj_k
end

@assert typestable(tau_to_wn, (FiniteTempBasisSet, Statistics, Array{ComplexF64,3}))
@assert typestable(wn_to_tau, (FiniteTempBasisSet, Statistics, Array{ComplexF64,3}))
