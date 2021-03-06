using LinearAlgebra, JLD, DelimitedFiles, KrylovKit
include("config.jl")

#default(size = (800,600), reuse = true)

crea(d) = diagm(-1 => [sqrt(i) for i=1:d-1])
anih(d) = Matrix(crea(d)')
numb(d) = crea(d)*anih(d)
disp(d) = crea(d)+anih(d)
mome(d) = crea(d)-anih(d)
sx = [0. 1.; 1. 0.]
sz = [1. 0.; 0. -1.]
sy = [0. -im; im 0.]
sp = [0. 1.; 0. 0.]
sm = Matrix(sp')

unitvec(n, d) = [fill(0, n-1)..., 1, fill(0, d-n)...]
unitmat(d1, d2) = [i1==i2 ? 1 : 0 for i1=1:d1, i2=1:d2]
unitmat(d) = unitmat(d, d)

function unitcol(n, d)
    z = zeros(d, 1)
    z[n] = 1
    return z
end

function unitrow(n, d)
    z = zeros(1, d)
    z[n] = 1
    return z
end

abstract type LightCone end

struct siteops
    h::Array{ComplexF64, 2}
    r::Array{ComplexF64, 2}
    l::Array{ComplexF64, 2}
end

mutable struct envops
    H::Array{ComplexF64, 2}
    op::Array{ComplexF64, 2}
    opd::Array{ComplexF64, 2}
end

mutable struct obbsite
    A::Array{ComplexF64, 3}
    V::Array{ComplexF64, 2}
    obbsite(A,V) = size(A,2) == size(V,2) ? new(A,V) : error("Dimension mismatch")
end

#chose n elements of x without replacement
function choose(x, n)
    len = length(x)
    if n > len
        throw(ArgumentError("cannot chose more elements than are contained in the list"))
        return
    end
    remaining = x
    res = Vector{eltype(x)}(undef, n)
    for i in 1:n
        rnd = rand(remaining)
        res[i] = rnd
        filter!(p -> p != rnd, remaining)
    end
    return res
end

"""
function therHam(psi, site1, site2)

calculates Hβ such that ρ = e^(-βH) for some density matrix ρ obatined from tracing out everything outside the range [site1,site2] in the mps psi
"""

function therHam(psi, site1, site2)
    pmat = ptracemps(psi, site1, site2)
    pmat = 0.5 * (pmat + pmat')
    Hb = -log(pmat)
    S = eigen(Hb).values
    return Hb, S
end

function loaddat(dir, unid, var)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"dat_",unid,".jld"), var)
end

function loaddat(dir, unid)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"dat_",unid,".jld"))
end
function loadconv(dir, unid, var)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"convdat_",unid,".jld"), var)
end

function loadconv(dir, unid)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"convdat_",unid,".jld"))
end

function savecsv(dir, fname, dat)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    writedlm(string(dir, fname,".csv"), dat, ',')
end

function eigenchain(cparams...; nummodes=nothing)
    if nummodes==nothing
        nummodes = length(cparams[1])
    end
    es=cparams[1][1:nummodes]
    ts=cparams[2][1:nummodes-1]
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    return eigen(hmat)
end

function thermaloccupations(β, cparams...)
    es=cparams[1]
    ts=cparams[2]
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U=F.vectors
    S=F.values
    ((U)^2) * (1 ./ (exp.(β.*S).-1))
end

function measuremodes(A, chainsection::Tuple{Int64,Int64}, e::Array{Float64,1}, t::Array{Float64,1}, c0=1)
    N=abs(chainsection[1]-chainsection[2])+1
    e=e[1:N]
    t=t[1:N-1]
    d=size(A[chainsection[1]])[2]#assumes constant number of Foch states
    bd=crea(d)
    b=anih(d)
    hmat = diagm(0=>e, 1=>t, -1=>t)
    F = eigen(hmat)
    U = F.vectors
    return real.(diag(U' * measure2siteoperator(A, bd, b, chainsection, hermitian=true) * U))
end
#for longer chains it can be worth calculating U in advance
function measuremodes(A, chainsection::Tuple{Int64,Int64}, U::AbstractArray)
    d=size(A[chainsection[1]])[2]#assumes constant number of Foch states
    bd=crea(d)
    b=anih(d)
    return real.(diag(U' * measure2siteoperator(A, bd, b, chainsection, hermitian=true) * U))
end

"""
function findchainlength(T, cparams...; eps=10^-6)

estimates length of chain required for a particular set of chain parameters by calulating how long an excitation on the first site takes to reach the end. The chain length is given as the length required for the excitation to have just reached the last site after time T.
"""

function findchainlength(T, cparams; eps=10^-4, verbose=false)
    Nmax = length(cparams[1])
    occprev = endsiteocc(T, [cparams[1][1:Nmax], cparams[2][1:Nmax-1]])
    occ = endsiteocc(T, [cparams[1][1:Nmax-1], cparams[2][1:Nmax-2]])

    verbose && println("truncating chain...")
    verbose && println("starting chain length $Nmax")

    if abs(occ-occprev) > eps
        throw(error("Suitable truncation not found, try larger starting chain length"))
    end
    occprev=occ
    for ntrunc=Nmax-2:-1:1
        verbose && println("ntrunc = $ntrunc")
        occ = endsiteocc(T, [cparams[1][1:ntrunc], cparams[2][1:ntrunc-1]])
        if abs(occ-occprev) > eps
            return ntrunc
        end
        occprev=occ
    end
end

"""
function chainprop(t, cparams...)

propagates an excitation placed initially on the first site of a tight-binding chain with parameters given by cparams for a time t and return occupation expectation for each site.
"""
function chainprop(t, cparams)
    es=cparams[1]
    ts=cparams[2]
    N=length(es)
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U = F.vectors
    S = F.values
    [real.(transpose(U[:,1].*exp.(im*t.*S))*U[:,i]*transpose(U[:,i])*(U[:,1].*exp.(-im*t.*S))) for i in 1:N]
end

function endsiteocc(t, cparams)
    es=cparams[1]
    ts=cparams[2]
    N=length(es)
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U = F.vectors
    S = F.values
    real.(transpose(U[:,1].*exp.(im*t.*S))*U[:,N]*transpose(U[:,N])*(U[:,1].*exp.(-im*t.*S)))
end

"""
function aniplot(dat; labels=nothing, fname="", title="", dir=ANIMDIR, endf=nothing)

Produces an animated plot of the data supplied by dat
"""
function aniplot(dat; xseries=nothing, xlabel="", ylabel="", label=nothing, title="", dir=ANIMDIR, endf=nothing)

    if dir[end] != '/'
        dir = string(dir,"/")
    end

    numplots = length(dat)

    pyplot(size = (800,600), legend = (label != nothing), reuse = true, title = title)
    len=length(dat[1])
    for i = 2:numplots
        if len != length(dat[i])
            throw(ArgumentError("data lists must have the same length"))
        end
    end

    endf = endf==nothing ? len : endf

    ymax=max(hcat([[max(dat[j][i]...) for i=1:len] for j=1:numplots]...)...)

    if label == nothing
        ani = @animate for i=1:endf
            for j = 1:numplots
                if xseries != nothing
                    scatter(xseries[j], dat[j][i], xlabel=xlabel, ylabel=ylabel, ylim=(0,ymax), markersize=5, c=[1])
                else
                    scatter(dat[j][i],xtickfontsize=18,ytickfontsize=18, xlabel=xlabel,xguidefontsize=30, ylabel=ylabel, yguidefontsize=30,ylim=(0,ymax), markersize=5, c=[1])
                end
            end
        end
    else
        ani = @animate for i=1:endf
            for j = 1:numplots
                if xseries != nothing
                    scatter(xseries[j], dat[j][i], xlabel=xlabel, ylabel=ylabel, ylim=(0,ymax), label=label[i], markersize=5, c=[1])
                else
                    scatter(dat[j][i], xlabel=xlabel, ylabel=ylabel, ylim=(0,ymax), label=label[j], markersize=5, c=[1])
                end
            end
        end
    end

    gif(ani, string(dir,"anim_",title,".gif"), fps=10)
end

function booltostr(b)
    return b ? "true" : "false"
end

function mean(list)
    n = length(list)
    return sum(list)/n
end

function var(list)
    n = length(list)
    m = mean(list)
    return sum((list .- m) .^ 2)/n
end

function sd(list)
    return sqrt(var(list))
end

"""
function msd(dat1::Vector{Float64}, dat2::Vector{Float64})

Calculates the root mean squared difference between two measurements of an observable over the same time period
"""
function rmsd(ob1, ob2)
    len = length(ob1)
    if len != length(ob2)
        throw(ArgumentError("inputs must have same length"))
    end
    return sqrt(sum((ob1 - ob2).^2)/len)
end

"""
function dynamap(ps1,ps2,ps3,ps4)

calulates complete dynamical map to time step at which ps1, ps2, ps3 and ps4 are specified
"""

function dynamap(ps1,ps2,ps3,ps4)
    #ps1 : time evolved system density matrix starting from initial state up
    #ps2 : time evolved system density matrix starting from initial state down
    #ps3 : time evolved system density matrix starting from initial state (up + down)/sqrt(2)
    #ps4 : time evolved system density matrix starting from initial state (up - i*down)/sqrt(2)
    ϵ = [
        ps1[1]-real(ps1[3])-imag(ps1[3]) ps2[1]-real(ps2[3])-imag(ps2[3]) ps3[1]-real(ps3[3])-imag(ps3[3]) ps4[1]-real(ps4[3])-imag(ps4[3]);
        ps1[4]-real(ps1[3])-imag(ps1[3]) ps2[4]-real(ps2[3])-imag(ps2[3]) ps3[4]-real(ps3[3])-imag(ps3[3]) ps4[4]-real(ps4[3])-imag(ps4[3]);
        2*real(ps1[3]) 2*real(ps2[3]) 2*real(ps3[3]) 2*real(ps4[3]);
        2*imag(ps1[3]) 2*imag(ps2[3]) 2*imag(ps3[3]) 2*imag(ps4[3])
    ]
    U = [1 (im-1)/2 -(im+1)/2 0; 0 (im-1)/2 -(im+1)/2 1; 0 1 1 0 ; 0 im -im 0]
    invU = inv(U)
    return invU*ϵ*U
end

function ttm2_calc(ps1, ps2, ps3, ps4)

    numsteps = length(ps1)

    ϵ = [dynamap(ps1[i],ps2[i],ps3[i],ps4[i]) for i=2:numsteps]

    T=[]

    for n in 1:numsteps-1
        dif=zero(ϵ[1])
        for m in 1:length(T)
            dif += T[n-m]*ϵ[m]
        end
        push!(T,ϵ[n] - dif)
    end
    return T
end

function ttm2_evolve(numsteps, T, ps0)

    K = length(T)
    ps = [reshape(ps0, 4, 1)]
    for i in 1:numsteps
        psnew = sum(T[1:min(i,K)] .* ps[end:-1:end-min(i,K)+1])
        ps = push!(ps, psnew)
    end
    return reshape.(ps, 2, 2)
end

function entropy(rho)
    λ = eigen(rho).values
    return real(sum(map(x-> x==0 ? 0 : -x*log(x), λ)))
end

function paramstring(x, sf)
    x = round(x, digits=sf)
    xstr = string(Int64(round(x*10^sf)))
    while length(xstr) < 2*sf
        xstr = string("0",xstr)
    end
    xstr
end

import LinearAlgebra: transpose
function transpose(A::AbstractArray, dim1::Int, dim2::Int)
    nd=ndims(A)
    perm=collect(1:nd)
    perm[dim1]=dim2
    perm[dim2]=dim1
    permutedims(A, perm)
end
function QR(A::AbstractArray, i::Int)
    dims = [size(A)...]
    nd = length(dims)
    ds = collect(1:nd)
    AL, C = qr(reshape(permutedims(A, circshift(ds, -i)), :, dims[i]))
    AL = permutedims(reshape(Matrix(AL), circshift(dims, -i)...), circshift(ds, i))
    return AL, C
end
function QR_full(A::AbstractArray, i::Int)
    dims = [size(A)...]
    nd = length(dims)
    ds = collect(1:nd)
    AL, C = qr(reshape(permutedims(A, circshift(ds, -i)), :, dims[i]))
    AL = permutedims(reshape(AL, circshift(dims, -i)[1:end-1]..., :), circshift(ds, i))
    return AL, C
end

"""
    randisometry([T=Float64], dims...)
Construct a random isometry
"""
randisometry(T, d1, d2) = d1 >= d2 ? Matrix(qr!(randn(T, d1, d2)).Q) : Matrix(lq!(randn(T, d1, d2)).Q)
randisometry(d1, d2) = randisometry(Float64, d1, d2)
randisometry(dims::Dims{2}) = randisometry(dims[1], dims[2])
randisometry(T, dims::Dims{2}) = randisometry(T, dims[1], dims[2])

"""
    U, S, Vd = svdtrunc(A; truncdim = max(size(A)...), truncerr = 0.)
Perform a truncated SVD, with maximum number of singular values to keep equal to `truncdim`
or truncating any singular values smaller than `truncerr`. If both options are provided, the
smallest number of singular values will be kept.
Unlike the SVD in Julia, this returns matrix U, a diagonal matrix (not a vector) S, and
Vt such that A ≈ U * S * Vt
"""
function svdtrunc(A; truncdim = max(size(A)...), truncerr = 0.)
    F = svd(A)
    d = min(truncdim, count(F.S .>= truncerr))
    return F.U[:,1:d], diagm(0=>F.S[1:d]), F.Vt[1:d, :]
end
