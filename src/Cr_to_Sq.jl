""" part of matrix A around the origin """
function pmat(
    A::Matrix{<:Number},
    Nx::Int=round(Int,size(A,1)/2),
    Ny::Int=round(Int,size(A,2)/2)
)
    [A[end-Nx+1:end,end-Ny+1:end] A[end-Nx+1:end,1:Ny+1];
     A[1:Nx+1,end-Ny+1:end] A[1:Nx+1,1:Ny+1]]
end

# -------------------------------------------
#       for plot
# -------------------------------------------
function _Cd_to_Sq(
    Cd::Array{ComplexF64,2},
    xrn::AbstractVector{<:Number},
    yrn::AbstractVector{<:Number},
    k1::Float64,
    k2::Float64;
    dmax::Float64=30.0
)
    k1*=sqrt(2/3)
    k2*=sqrt(2)
    kx=cospi(0.25)k1-sinpi(0.25)k2
    ky=sinpi(0.25)k1+cospi(0.25)k2
    # kx=k1; ky=k2

    s=0.0im
    nx,ny=size(Cd)
    for iy in 1:ny, ix in 1:nx
        sqrt(xrn[ix]^2+yrn[iy]^2)>dmax+1e-10 && continue
        s+=Cd[ix,iy]*cis(kx*xrn[ix]+ky*yrn[iy])
    end
    s
end

function Cd_to_Sq(
    Cd::Array{ComplexF64,2},
    k::AbstractArray{<:Number,2};
    dmax::Float64=30.0
)
    nx,ny=size(Cd)
    nx=round(Int,(nx-1)/2)
    ny=round(Int,(ny-1)/2)
    xrn=-nx:nx
    yrn=-ny:ny

    nk=size(k,2)
    Sq=Array{ComplexF64}(undef,nk)
    Threads.@threads for ii in 1:nk
        Sq[ii] = _Cd_to_Sq(Cd,xrn,yrn,k[1,ii],k[2,ii];dmax=dmax)
    end
    Sq
end

function Cd_to_Sq(
    Cd::Array{ComplexF64,2},
    k1::AbstractArray{<:Number,1},
    k2::AbstractArray{<:Number,1};
    dmax::Float64=30.0
)
    nx,ny=map(x->round(Int,(x-1)/2),size(Cd))
    xrn=-nx:nx
    yrn=-ny:ny

    nk1=length(k1)
    nk2=length(k2)
    Sq=Array{ComplexF64}(undef,nk1,nk2)
    Threads.@threads for i2 in 1:nk2
        for i1 in 1:nk1
            Sq[i1,i2] = _Cd_to_Sq(Cd,xrn,yrn,k1[i1],k2[i2];dmax=dmax)
        end
    end
    Sq
end