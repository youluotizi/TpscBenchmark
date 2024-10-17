#=
using CairoMakie
set_theme!(
    size=(520,400),
    markersize=12,
    linewidth=2.5,
    fontsize = 20
)

begin
    rn=LinRange(0,15,100)
    x=[ix for ix in rn,iy in rn]
    y=[ix+iy for ix in rn,iy in rn]
    z=map((x,y)->cos(x)*sin(y), x, y)
end
surface(x,y,z; shading=false,
    axis=(;type=Axis3, azimuth=-0.5pi, elevation=0.5pi,
    zticklabelsvisible=false,
    zlabelvisible=false
    )
)
=#

function mylines(x,ylist; kargs...)
    Ny=size(ylist,2)
    f,ax,l1=lines(x,ylist[:,1]; kargs...)
    for ii in 2:Ny
        lines!(ax,x,ylist[:,ii])
    end
    f
end

function myhmap(data...; ax=(;aspect=1),fig=(;size=(480,400)),args...)
    f,_,hm=heatmap(data...; axis=ax, figure=fig, colormap=:bluesreds,args...)
    Colorbar(f[1,2],hm)
    f
end