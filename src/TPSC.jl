module TPSC
using LinearAlgebra #, MKL
using FFTW
using Roots
using SparseIR
import SparseIR: Statistics, valueim
# using Optim
   
export FiniteTempBasisSet, TPSCSolver, solve!, solve_DQMC!, cal_ek,pmat,spin_vertex_fit_line!,cal_docc!

include("fourier.jl")
include("cal_TPSC.jl")
include("Cr_to_Sq.jl")
include("solve.jl")

# include("TPSCplus.jl")

end