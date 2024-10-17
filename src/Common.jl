module Common
using LinearAlgebra

export LinBz,bz2d,norBev!,gaugev!,hinv,Schmidt,ArrayCutOff,myfilter,expshow

# 沿高对称线离散化,用于计算能带
struct LinBz
    Nk::Int
    hpt::Array{Int,1}
    rr::Array{Float64,1}
    kk::Array{Float64,2}
end
function LinBz(plist::Array{Array{Float64,1},1},num::Int)
    Nlist=length(plist)
    if Nlist>num
        num=Nlist
        println("num input error")
    end
    point=Array{Float64,2}(undef,2,Nlist) # 高对称点
    for ii=1:Nlist
        point[:,ii].=plist[ii]
    end
    Npath=Nlist-1
    totalpath=0.0 # 总长度
    path=Vector{Float64}(undef,Npath) # 每段折线的长度
    Vpath=Array{Float64,2}(undef,2,Npath)
    for ii=1:Npath
        Vpath[1,ii]=point[1,ii+1]-point[1,ii]
        Vpath[2,ii]=point[2,ii+1]-point[2,ii]
        path[ii]=sqrt(Vpath[1,ii]^2+Vpath[2,ii]^2)
        abs(path[ii])<1e-8 && println("Kpoint input error")
        totalpath+=path[ii]
    end
    num2=num-Nlist # 除了高对称点以外的点的数目
    pathNpoint=Vector{Int}(undef,Npath) # 每段折线分配的点数
    dpath=Array{Float64,1}(undef,Npath) # 每段折线划分的长度
    for ii=1:Npath
        pathNpoint[ii]=floor(Int,num2*path[ii]/totalpath) # 
        dpath[ii]=path[ii]/(pathNpoint[ii]+1)
    end
    for ii=1:Npath
        num2-=pathNpoint[ii] # 按长度划分后剩下的点
    end
    while num2>0 # 剩下的点按 dpath 长度继续分配
        _,pmax=findmax(dpath) 
        pathNpoint[pmax]+=1
        dpath[pmax]=path[pmax]/(pathNpoint[pmax]+1)
        num2-=1
    end
    sum(pathNpoint)+Nlist!=num && println("Kpoint partition error")

    kk=Array{Float64,2}(undef,2,num)
    ik=0
    for ii=1:Npath
        Numtmp=pathNpoint[ii]+2
        xrange=range(point[1,ii],point[1,ii+1],length=Numtmp)
        yrange=range(point[2,ii],point[2,ii+1],length=Numtmp)
        for (ix,iy) in zip(xrange,yrange)
            ik+=1
            kk[1,ik]=ix
            kk[2,ik]=iy
        end
        ik-=1
    end

    rr=Vector{Float64}(undef,num) # 路程
    rr[1]=0.0
    ik=1
    for ii=1:Npath
        for _ in 1:pathNpoint[ii]+1
            rr[ik+1]=rr[ik]+dpath[ii]
            ik+=1
        end
    end
    hpt=Vector{Int}(undef,Nlist) # 高对称点点位置
    hpt[1]=1
    for ii=1:Npath
        hpt[ii+1]=hpt[ii]+pathNpoint[ii]+1
    end

    LinBz(num,hpt,rr,kk)
end
function LinBz(kpoint::Array{Array{Float64,1},1})   
    Nk=length(kpoint)
    hpt=collect(1:Nk)
    rr=Array{Float64,1}(undef,Nk)
    kk=Array{Float64,2}(undef,2,Nk)

    rr[1]=0.0
    kk[:,1].=kpoint[1]
    tmp=Array{Float64,1}(undef,2)
    for ii=2:Nk
        tmp.=kpoint[ii].-kpoint[ii-1]
        rr[ii]=rr[ii-1]+sqrt(tmp[1]^2+tmp[2]^2)
        kk[:,ii].=kpoint[ii]
    end
    LinBz(Nk,hpt,rr,kk)
end

# 二维离散化
function bz2d(plist::Array{Array{Float64,1},1},nn::Array{Int,1})
    p=plist[1]
    b1=plist[2]./nn[1]
    b1=plist[3]./nn[2]
    bz=Array{Float64}(undef,2,nn[1]+1,nn[2]+1)
    for jj in 0:nn[2],ii in 0:nn[1]
        bz[:,ii+1,jj+1].=p.+jj.*b2.+ii.*b1
    end
    bz
end

# BdG 归一化
function norBev!(bv::AbstractMatrix{<:Number})
    Nv=size(bv,2)
    lenv=round(Int,size(bv,1)/2)
    for ii=1:Nv
        tmp=0.0
        for jj=1:lenv
            tmp+=abs2(bv[jj,ii])
            tmp-=abs2(bv[jj+lenv,ii])
        end
        tmp=abs(tmp)
        if tmp<5e-6
            for jj=1:2*lenv
                bv[jj,ii]*=√2
            end
            continue
        end
        tmp=1.0/sqrt(tmp)
        for jj=1:2*lenv
            bv[jj,ii]*=tmp
        end
    end
    nothing
end

# 规范，取本征向量中模最大的分量相位为零
function gaugev!(ev::Array{T}) where T<:Union{ComplexF64,Float64}
    d=size(ev)
    Nv=1
    if length(d)!=1
        for ii in 2:length(d)
            Nv*=d[ii]
        end
    end

    for ii in 1:Nv
        p1=(ii-1)*d[1]+1
        p2=p1+d[1]-1
        pt=p1
        mpt=abs(ev[pt])+1e-6
        for jj in p1:p2    
            if abs(ev[jj])>mpt
                pt=jj
                mpt=abs(ev[jj])+1e-6
            end
        end

        phs::T=abs(ev[pt])/ev[pt]
        for jj in p1:p2
            ev[jj]*=phs
        end
    end
    nothing
end

# 厄米矩阵求逆
function hinv(v::Hermitian{T,Matrix{T}},p::Float64=1e-9) where T<:Number
    en,ev=eigen(v)
    en2=similar(en)
    ev2=Array{T,2}(undef,length(en),length(en))
    for ii in eachindex(en)
        if abs(en[ii])>p
            en2[ii]=1/en[ii]
        else
            println("hinv error")
            ev2.=NaN
            return ev2
        end
    end
    ev2.=ev*Diagonal(en2)*ev'
    return ev2
end

# 斯密特正交化
function Schmidt(v::Matrix{T}) where T
    lv,nv=size(v)
    vo=Matrix{T}(undef,lv,nv)
    vtmp=Array{T}(undef,lv)
    nonzero=0
    for ii=1:nv
        if norm(v[:,ii])>1e-8
            vo[:,ii].=normalize(v[:,ii])
            nonzero=ii
            break
        end
        vo[:,ii].=v[:,ii]
    end
    for ii=nonzero+1:nv
        if norm(v[:,ii])<1e-8
            vo[:,ii].=v[:,ii]
            continue
        end
        vtmp.=v[:,ii]
        for jj=1:ii-1
            vtmp.-=(vo[:,jj]'*v[:,ii]).*vo[:,jj]
        end
        if norm(vtmp)<1e-8
            vo[:,ii].=vtmp
            continue
        end
        vo[:,ii].=normalize(vtmp)
    end
    # nn=Array{Float64,1}(undef,nv)
    # for ii =1:nv
    #     nn[ii]=norm(vo[:,ii])
    # end
    # pt=sortperm(nn,rev=true)
    # vo.=vo[:,pt]
    return vo
end

# 求和，当出现NaN时就跳过
function sumNaN(a::Array{T}) where T<:Number
    tmp=zero(T)
    for ii in a
        isnan(ii) && continue
        tmp+=ii
    end
    return tmp
end

# 数组截断
function ArrayCutOff(v::Array{Float64},cutdown::Float64,cutup::Float64)
    v1=similar(v)
    for ii in eachindex(v)
        if v[ii]<cutdown
            v1[ii]=cutdown
        elseif v[ii]>cutup
            v1[ii]=cutup
        else
            v1[ii]=v[ii]
        end
    end
    return v1
end

# 
function myfilter(x::Float64,p::Float64=1e-8;digit=4)
    abs(x)<p ? 0.0 : round(x,sigdigits=digit)
end

function myfilter(x::ComplexF64,p::Float64=1e-8;digit=4)
    r1=myfilter(real(x),p;digit=digit)
    r2=myfilter(imag(x),p;digit=digit)
    return complex(r1,r2)
end

function myfilter(v::Array{<:Number},p::Float64=1e-8;digit=4)
    v1=similar(v)
    for ii in eachindex(v)
        v1[ii]=myfilter(v[ii],p;digit=digit)
    end
    return v1
end

function expshow(a::Array{ComplexF64})
    b=[(myfilter(abs(ii)),myfilter(angle(ii))/1pi) for ii in a]
    return b
end
function expshow(a::ComplexF64)
    (myfilter(abs(a)),myfilter(angle(a)/1pi))
end


end

