using TensorOperations
include("config.jl")
include("fundamentals.jl")
include("tensorOps.jl")

mutable struct ChainLightCone <: LightCone
    ref::Vector # vector of OCs of initial mps
    edge::Int # last site to be ev
    thresh::Float64
    ChainLightCone(A::Vector, edge::Int, thresh::Float64) = new(orthcentersmps(A), edge, thresh)
end
ChainLightCone(A::Vector) = ChainLightCone(A, 2, DEFLCTHRESH)
ChainLightCone(A::Vector, rad::Int) = ChainLightCone(A, rad, DEFLCTHRESH)
LightCone(A::Vector) = ChainLightCone(A, 2, DEFLCTHRESH)
LightCone(A::Vector, rad::Int) = ChainLightCone(A, rad, DEFLCTHRESH)

#returns list of the orthoganality centres of A, assumes A is right-normalised
function orthcentersmps(A::Vector)
    B = deepcopy(A)
    N = length(B)
    for i in 2:N
        AL, C = QR(B[i-1], 2)
        @tensor AC[:] := C[-1,1] * B[i][1,-2,-3]
        B[i] = AC
    end
    return B
end

function initenvs(A::Vector, M::Vector, F::Nothing)
    N = length(A)
    F = Vector{Any}(undef, N+2)
    F[1] = fill!(similar(M[1], (1,1,1)), 1)
    F[N+2] = fill!(similar(M[1], (1,1,1)), 1)
    for k = N:-1:1
        F[k+1] = updaterightenv(A[k], M[k], F[k+2])
    end
    return F
end
initenvs(A::Vector, M::Vector) = initenvs(A, M, nothing)
function initenvs(A::Vector, M::Vector, F::Vector)
    return F
end

function tdvp1sweep!(dt2, A::Vector, M::Vector, F=nothing; verbose=false, kwargs...)

    N = length(A)

    dt = dt2/2
    F = initenvs(A, M, F)

    AC = A[1]
    for k = 1:N-1
        AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(AC, applyH1(AC, M[k], F[k], F[k+2])))
            println("Sweep L->R: AC site $k, energy $E")
        end

        AL, C = QR(AC, 2)
        A[k] = AL
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        C, info = exponentiate(x->applyH0(x,F[k+1],F[k+2]), im*dt, C; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(C, applyH0(C, F[k+1], F[k+2])))
            println("Sweep L->R: C between site $k and $(k+1), energy $E")
        end
        @tensor AC[:] := C[-1,1] * A[k+1][1,-2,-3]

    end
    k = N
    AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -2*im*dt, AC; ishermitian = true, kwargs...)
    if verbose
        E = real(dot(AC, applyH1(AC, M[k], F[k], F[k+2])))
        println("Sweep L->R: AC site $k, energy $E")
    end

    for k = N-1:-1:1
        AR, C = QR(AC, 1)
        A[k+1] = AR
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        C, info = exponentiate(x->applyH0(x, F[k+2], F[k+1]), im*dt, C; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(C, applyH0(C, F[k+2], F[k+1])))
            println("Sweep R->L: C between site $k and $(k+1), energy $E")
        end
        @tensor AC[:] := C[-2,1] * A[k][-1,1,-3]

        AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(AC, applyH1(AC, M[k], F[k], F[k+2])))
            println("Sweep R->L: AC site $k, energy $E")
        end
    end
    A[1] = AC
    return A, F
end
tdvp1sweep!(dt2, A::Vector, M::Vector, F, lc::Nothing; verbose=false, kwargs...) = tdvp1sweep!(dt2, A, M, F; verbose=verbose, kwargs...)
tdvp1sweep!(dt2, A::Vector, M::Vector, lc::ChainLightCone; verbose=false, kwargs...) = tdvp1sweep!(dt2, A, M, nothing, lc::ChainLightCone; verbose=verbose, kwargs...)

function tdvp1sweep!(dt2, A::Vector, M::Vector, F, lc::ChainLightCone; verbose=false, kwargs...)

    N = length(A)

    dt = dt2/2
    F = initenvs(A, M, F)

    k=1
    AC = A[k]
    AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt, AC; ishermitian = true, kwargs...)
    if verbose
        E = real(dot(AC, applyH1(AC, M[k], F[k], F[k+2])))
        println("Sweep L->R: AC site 1, energy $E")
    end

    if k == lc.edge && lc.edge != N
        @tensor v = scalar(conj(AC[a,b,s])*lc.ref[k+1][a,b,s])
        if 1-norm(v) > lc.thresh
            lc.edge += 1
        end
    end

    while k <= lc.edge-1 && k <= N-1

        AL, C = QR(AC, 2)
        A[k] = AL
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        C, info = exponentiate(x->applyH0(x,F[k+1],F[k+2]), im*dt, C; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(C, applyH0(C, F[k+1], F[k+2])))
            println("Sweep L->R: C between site $k and $(k+1), energy $E")
        end

        @tensor AC[:] := C[-1,1] * A[k+1][1,-2,-3]

        AC, info = exponentiate(x->applyH1(x, M[k+1], F[k+1], F[k+3]), -im*dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(AC, applyH1(AC, M[k+1], F[k+1], F[k+3])))
            println("Sweep L->R: AC site $k, energy $E")
        end

        if k == lc.edge-1 && lc.edge != N
            @tensor v = scalar(conj(AC[a,b,s])*lc.ref[k+1][a,b,s])
            if 1-norm(v) > lc.thresh
                lc.edge += 1
            end
        end
        k+=1
    end

    for k = lc.edge-1:-1:1

        AC, info = exponentiate(x->applyH1(x, M[k+1], F[k+1], F[k+3]), -im*dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(AC, applyH1(AC, M[k+1], F[k+1], F[k+3])))
            println("Sweep R->L: AC site $k, energy $E")
        end

        AR, C = QR(AC, 1)
        A[k+1] = AR
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        C, info = exponentiate(x->applyH0(x, F[k+2], F[k+1]), im*dt, C; ishermitian = true, kwargs...)
        if verbose
            E = real(dot(C, applyH0(C, F[k+2], F[k+1])))
            println("Sweep R->L: C between site $k and $(k+1), energy $E")
        end
        @tensor AC[:] := C[-2,1] * A[k][-1,1,-3]

    end

    AC, info = exponentiate(x->applyH1(x, M[1], F[1], F[3]), -im*dt, AC; ishermitian = true, kwargs...)
    if verbose
        E = real(dot(AC, applyH1(AC, M[1], F[1], F[3])))
        println("Sweep R->L: AC site 1, energy $E")
    end

    A[1] = AC
    return A, F
end

function normmps(A::Vector; right=false, left=false, cen=nothing)
    N = length(A)
    if right
        @tensor n[a',b'] := A[1][a',c,s]*conj(A[1][b',c,s])
        return n[1,1]
    elseif left
        @tensor n[a',b'] := A[N][a,a',s]*conj(A[N][a,b',s])
        return n[1,1]
    elseif typeof(cen)==Int
        @tensor n = A[cen][a,b,s]*conj(A[cen][a,b,s])
        return n
    else
        ρ = ones(eltype(A[1]), 1, 1)
        for k=1:N
            @tensor ρ[a,b] := ρ[a',b']*A[k][b',b,s]*conj(A[k][a',a,s])
        end
        return ρ[1,1]
    end
end

function mpsrightnorm!(A::Vector, jq=nothing)
    nsites = length(A)
    if jq==nothing
        jq=1
    end
    for i=nsites:-1:jq+1
        aleft, aright, aup = size(A[i])
        C, AR = lq(reshape(A[i], aleft, aright*aup))
        A[i] = reshape(Matrix(AR), aleft, aright, aup)
        @tensor AC[:] := A[i-1][-1,1,-3] * C[1,-2]
        A[i-1] = AC
    end
end
function mpsleftnorm!(A::Vector{T}, jq=nothing) where {T}
    nsites = length(A)
    if jq==nothing
        jq=nsites
    end
    for i=1:jq-1
        AL, C = QR(A[i], 2)
        A[i] = AL
        @tensor AC[:] := C[-1,1] * A[i+1][1,-2,-3]
        A[i+1] = AC
    end
end
function mpsmixednorm!(A::Vector, centre::Int)
    mpsleftnorm!(A, centre)
    mpsrightnorm!(A, centre)
end

function randmps(physdims::Dims{N}, Dmax::Int, T::Type{<:Number} = Float64) where {N}
    bonddims = Vector{Int}(undef, N+1)
    bonddims[1] = 1
    bonddims[N+1] = 1
    Nhalf = div(N,2)
    for n = 2:N
        bonddims[n] = min(Dmax, bonddims[n-1]*physdims[n-1])
    end
    for n = N:-1:1
        bonddims[n] = min(bonddims[n], bonddims[n+1]*physdims[n])
    end

    As = Vector{Any}(undef, N)
    for n = 1:N
        d = physdims[n]
        Dl = bonddims[n]
        Dr = bonddims[n+1]
        As[n] = reshape(randisometry(T, Dl, Dr*d), (Dl, Dr, d))
    end
    return As
end
randmps(N::Int, d::Int, Dmax::Int, T = Float64) = randmps(ntuple(n->d, N), Dmax, T)

function productstatemps(physdims::Dims, Dmax::Int=1; statelist=nothing)

    N = length(physdims)

    if typeof(Dmax)<:Number
        Dmax=fill(Dmax, N-1)
    end

    bonddims = Vector{Int}(undef, N+1)
    bonddims[1] = 1
    bonddims[N+1] = 1

    if statelist == nothing
        statelist = [unitcol(1, physdims[i]) for i in 1:N]
    end

    for i=2:N
        bonddims[i] = min(Dmax[i-1], bonddims[i-1] * physdims[i-1])
    end
    for i=N:-1:2
        bonddims[i] = min(bonddims[i], bonddims[i+1] * physdims[i])
    end

    B0 = Vector{Any}(undef, N)
    for i=1:N
        A = zeros(ComplexF64, bonddims[i], bonddims[i+1], physdims[i])
        for j=1:min(bonddims[i], bonddims[i+1])
            A[j,j,:] = statelist[i]
        end
        B0[i] = A
    end

    mpsrightnorm!(B0)
    return B0
end
productstatemps(N::Int, d::Int, Dmax::Int=1; statelist=nothing) = productstatemps(ntuple(i->d, N), Dmax; statelist=statelist)

"""
    entanglemententropy(A)
For a list of tensors `A` representing a right orthonormalized MPS, compute the entanglement
entropy for a bipartite cut for every bond
"""
#If you want the entanglements at every point during an evolution it would be more efficient to
#replace the QRs with SVDs in the tdvp routine
function entanglemententropy(A)
    N = length(A)
    entropy = Vector{Float64}(undef, N-1)

    A1 = A[1]
    aleft, aright, aup = size(A1)
    U, S, V = svdtrunc(reshape(permutedims(A1, [1,3,2]), aleft*aup, :))
    schmidt = diag(S)
    entropy[1] = sum([schmidt[i]==0 ? 0 : -schmidt[i]^2 * log(schmidt[i]^2) for i=1:length(schmidt)])
    for k = 2:N-1
        Ak = A[k]
        aleft, aright, aup = size(Ak)
        Ak = reshape(S*V*reshape(Ak, aleft, :), aleft, aright, aup)
        U, S, V = svdtrunc(reshape(permutedims(Ak, [1,3,2]), aleft*aup, :))
        schmidt = diag(S)
        entropy[k] = sum([schmidt[i]==0 ? 0 : -schmidt[i]^2 * log(schmidt[i]^2) for i=1:length(schmidt)])
    end
    return entropy
end

"""
function svdmps(A)

For a right normalised mps computes the full svd spectrum for a bipartition at every bond
"""
function svdmps(A)
    N = length(A)
    spec = Vector{Any}(undef, N-1)
    A1 = A[1]
    aleft, aright, aup = size(A1)
    U, S, V = svdtrunc(reshape(permutedims(A1, [1,3,2]), aleft*aup, :))
    spec[1] = diag(S)
    for k = 2:N-1
        Ak = A[k]
        aleft, aright, aup = size(Ak)
        Ak = reshape(S*V*reshape(Ak, aleft, :), aleft, aright, aup)
        U, S, V = svdtrunc(reshape(permutedims(Ak, [1,3,2]), aleft*aup, :))
        spec[k] = diag(S)
    end
    return spec
end

function measurempo(A::Vector, M::Vector)
    N = length(A)
    F = ones(ComplexF64, 1, 1, 1)
    for k=1:N
        @tensor F[a',b',c'] := F[a,b,c]*A[k][c,c',d]*M[k][b,b',e,d]*conj(A[k][a,a',e])
    end
    real(F[1,1,1])
end

function elementmps(A, el...)
    nsites = length(A)
    c = A[1][:,:,el[1]]
    for i=2:nsites
        c = c * A[i][:,:,el[i]]
    end
    c[1,1]
end

function elementmpo(M, el...)
    nsites = length(M)
    c = M[1][:,:,el[1][1],el[1][2]]
    for i=2:nsites
        c *= M[i][:,:,el[i][1],el[i][2]]
    end
    c[1,1]
end

function apply1siteoperator!(A, O, sites::Vector{Int})
    for i in sites
        @tensor R[a,b,s] := O[s,s']*A[i][a,b,s']
        A[i] = R
    end
end
apply1siteoperator!(A, O, site::Int) = apply1siteoperator!(A, O, [site])

"""
    measure1siteoperator(A, O)
For a list of tensors `A` representing a right orthonormalized MPS, compute the local expectation
value of a one-site operator O for every site or just one if i is specified.

For calculating operators on single sites this will be more efficient if the site is on the left of the mps.
"""

function measure1siteoperator(A::Vector, O, sites::Vector{Int})
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = zeros(ComplexF64, N)
    for i=1:N
        if in(i, sites)
            @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*O[s',s]*conj(A[i][a,c,s']))
            expval[i] = v
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    return expval[sites]
end
function measure1siteoperator(A::Vector, O, chainsection::Tuple{Int64,Int64})
    ρ = ones(ComplexF64, 1, 1)
    l=min(chainsection...)#leftmost site in section
    r=max(chainsection...)#rightmost site in section
    rev = l != chainsection[1]
    len = r-l+1
    expval = zeros(ComplexF64, len)

    for i=1:l-1
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    for i=l:r-1
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*O[s',s]*conj(A[i][a,c,s']))
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
        expval[i-l+1] = v
    end
    @tensor v = scalar(ρ[a,b]*A[r][b,c,s]*O[s',s]*conj(A[r][a,c,s']))
    expval[len] = v
    if rev
        reverse!(expval)
    end
    return expval
end
function measure1siteoperator(A::Vector, O)
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = Vector{ComplexF64}()
    for i=1:N
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*O[s',s]*conj(A[i][a,c,s']))
        push!(expval, v)
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    return expval
end
function measure1siteoperator(A::Vector, O, site::Int)
    ρ = ones(ComplexF64, 1, 1)
    for i=1:site-1
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    @tensor v = scalar(ρ[a,b]*A[site][b,c,s]*O[s',s]*conj(A[site][a,c,s']))
    return v
end

"""
    measure2siteoperator(A, M1, M2, ...)
Gives expectation of M1*M2 where M1 acts on site j1 and M2 acts on site j2
assumes A is right normalised
"""

function measure2siteoperator(A::Vector, M1, M2, j1::Int64, j2::Int64)
    ρ = ones(ComplexF64, 1, 1)
    i1=min(j1,j2)
    i2=max(j1,j2)
    m1 = j1<j2 ? M1 : M2
    m2 = j1<j2 ? M2 : M1
    if j1==j2
        return measure1siteoperator(A, M1*M2, j1)
    end
    for k=1:i1-1
        @tensor ρ[a',b'] := ρ[a,b]*A[k][b,b',s]*conj(A[k][a,a',s])
    end
    @tensor ρ[a',b'] := ρ[a,b]*A[i1][b,b',s]*m1[s',s]*conj(A[i1][a,a',s'])
    for k=i1+1:i2-1
        @tensor ρ[a',b'] := ρ[a,b]*A[k][b,b',s]*conj(A[k][a,a',s])
    end
    @tensor v = scalar(ρ[a,b]*A[i2][b,a',s]*m2[s',s]*conj(A[i2][a,a',s']))
    return v
end
function measure2siteoperator(A::Vector, M1, M2; hermitian=false)
    if hermitian
        return measure2siteoperator_herm(A, M1, M2)
    end
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = zeros(ComplexF64, N, N)
    for i in 1:N
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
        expval[i,i] = v
        @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
        @tensor ρ21[a,b] := ρ[a',b']*A[i][b',b,s]*M2[s',s]*conj(A[i][a',a,s'])
        for j in i+1:N
            @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
            expval[i,j] = v
            @tensor v = scalar(ρ21[a,b]*A[j][b,c,s]*M1[s',s]*conj(A[j][a,c,s']))
            expval[j,i] = v
            @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
            @tensor ρ21[a,b] := ρ21[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    return expval
end
function measure2siteoperator_herm(A::Vector, M1, M2)
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = zeros(ComplexF64, N, N)
    for i in 1:N
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
        expval[i,i] = v
        @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
        for j in i+1:N
            @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
            expval[i,j] = v
            @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    return expval + (expval' - diagm(0 => diag(expval)))
end
function measure2siteoperator(A::Vector, M1, M2, sites1::Vector{Int}, sites2::Vector{Int}; hermitian=false)
    if hermitian
        return measure2siteoperator_herm(A, M1, M2, sites1, sites2)
    end
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = zeros(ComplexF64, N, N)
    for i in 1:N
        if in(i, sites1)
            if in(i, sites2)
                @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
                expval[i,i] = v
            end
            @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
            for j in i+1:N
                if in(j, sites2)
                    @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
                    expval[i,j] = v
                end
                @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
            end
        end

        if in(i, sites2)
            @tensor ρ21[a,b] := ρ[a',b']*A[i][b',b,s]*M2[s',s]*conj(A[i][a',a,s'])
            for j in i+1:N
                if in(j, sites1)
                    @tensor v = scalar(ρ21[a,b]*A[j][b,c,s]*M1[s',s]*conj(A[j][a,c,s']))
                    expval[j,i] = v
                end
                @tensor ρ21[a,b] := ρ21[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
            end
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    return expval[sites1,sites2]
end
function measure2siteoperator_herm(A::Vector, M1, M2, sites1::Vector{Int}, sites2::Vector{Int})
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)
    expval = zeros(ComplexF64, N, N)
    for i in 1:N
        if in(i, sites1)
            if in(i, sites2)
                @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
                expval[i,i] = v
            end
            @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
            for j in i+1:N
                if in(j, sites2)
                    @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
                    expval[i,j] = v
                end
                @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
            end
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    expval = expval[sites1, sites2]
    expval = expval + (expval' - diagm(0 => diag(expval)))
    return expval
end
function measure2siteoperator(A::Vector, M1, M2, chainsection::Tuple{Int,Int}; hermitian=false)
    if hermitian
        return measure2siteoperator_herm(A, M1, M2, chainsection)
    end
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)

    l=min(chainsection...)#leftmost site in section
    r=max(chainsection...)#rightmost site in section
    rev = l != chainsection[1]
    len = r-l+1
    expval = zeros(ComplexF64, len, len)

    for i in 1:l-1
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    for i in l:r
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
        expval[i-l+1,i-l+1] = v
        @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
        @tensor ρ21[a,b] := ρ[a',b']*A[i][b',b,s]*M2[s',s]*conj(A[i][a',a,s'])
        for j in i+1:r
            @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
            expval[i-l+1,j-l+1] = v
            @tensor v = scalar(ρ21[a,b]*A[j][b,c,s]*M1[s',s]*conj(A[j][a,c,s']))
            expval[j-l+1,i-l+1] = v
            @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
            @tensor ρ21[a,b] := ρ21[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    if rev
        expval = reverse(reverse(expval, dims=1), dims=2)
    end
    return expval
end
function measure2siteoperator_herm(A::Vector, M1, M2, chainsection::Tuple{Int64,Int64})
    N = length(A)
    ρ = ones(ComplexF64, 1, 1)

    l=min(chainsection...)#leftmost site in section
    r=max(chainsection...)#rightmost site in section
    rev = l != chainsection[1]
    len = r-l+1
    expval = zeros(ComplexF64, len, len)

    for i in 1:l-1
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    for i in l:r
        @tensor v = scalar(ρ[a,b]*A[i][b,c,s]*M1[s',s'']*M2[s'',s]*conj(A[i][a,c,s']))
        expval[i-l+1,i-l+1] = v
        @tensor ρ12[a,b] := ρ[a',b']*A[i][b',b,s]*M1[s',s]*conj(A[i][a',a,s'])
        for j in i+1:r
            @tensor v = scalar(ρ12[a,b]*A[j][b,c,s]*M2[s',s]*conj(A[j][a,c,s']))
            expval[i-l+1,j-l+1] = v
            @tensor ρ12[a,b] := ρ12[a',b']*A[j][b',b,s]*conj(A[j][a',a,s])
        end
        @tensor ρ[a,b] := ρ[a',b']*A[i][b',b,s]*conj(A[i][a',a,s])
    end
    expval = expval + (expval' - diagm(0 => diag(expval)))
    if rev
        expval = reverse(reverse(expval, dims=1), dims=2)
    end
    return expval
end

function reversemps!(A)
    N = length(A)
    reverse!(A)
    for i=1:N
        A[i] = permutedims(A[i], [2,1,3])
    end
end

function reversempo!(M)
    N = length(M)
    reverse!(M)
    for i=1:N
        M[i] = permutedims(M[i], [2,1,3,4])
    end
end

#returns a list of the local Hilbert space dimensions of an MPO M
function physdims(M::Vector)
    N = length(M)
    res = Vector{Int}(undef, N)
    for (i, site) in enumerate(M)
        res[i] = size(site)[end]
    end
    return Dims(res)
end

function bonddimsmps(A::Vector)
    N=length(A)
    res = Vector{Int}(undef, N+1)
    res[1] = 1
    res[end] = 1
    for i=2:N
        res[i] = size(A[i])[1]
    end
    return Dims(res)
end

function calcXY(A_::Vector, H::Vector)
    A=deepcopy(A_)
    N = length(A)
    F = Vector{Any}(undef, N+2)
    F[1] = fill!(similar(H[1], (1,1,1)), 1)
    F[N+2] = fill!(similar(H[1], (1,1,1)), 1)

    for k = N:-1:1
        D1, D2, d = size(A[k])
        ARfull, C = QR_full(A[k], 1)
        FR = F[k+2][1:D2,:,:]
        AR = ARfull[1:D1,:,:]
        @tensor F[k+1][a,b,c] := FR[a1,b1,c1]*conj(ARfull[a,a1,s'])*H[k][b,b1,s',s]*AR[c,c1,s]
        if k!=1
            @tensor A[k-1][a,b,s] := A[k-1][a,b',s] * C[b,b']
        end
    end
    E = real(F[2][1])

    X = Vector{Any}(undef, N)
    Y = Vector{Any}(undef, N)
    AC = A[1]
    for k=1:N-1
        X[k] = applyH1(AC, H[k], F[k], F[k+2])
        D1, D2, d = size(AC)
        ALfull, C = QR_full(AC, 2)
        FL = F[k][1:D1,:,:]
        AL = ALfull[:,1:D2,:]
        @tensor F[k+1][a,b,c] := FL[a0,b0,c0]*conj(ALfull[a0,a,s'])*H[k][b0,b,s',s]*AL[c0,c,s]
        Y[k] = applyH0(C, F[k+1], F[k+2])
        @tensor AC[a,b,s] := C[a,a'] * A[k+1][a',b,s]
    end
    k=N
    X[k] = applyH1(AC, H[k], F[k], F[k+2])
    D1, D2, d = size(AC)
    ALfull, C = QR_full(AC, 2)
    FL = F[k][1:D1,:,:]
    AL = ALfull[:,1:D2,:]
    @tensor F[k+1][a,b,c] := FL[a0,b0,c0]*conj(ALfull[a0,a,s'])*H[k][b0,b,s',s]*AL[c0,c,s]
    Y[k] = applyH0(C, F[k+1], F[k+2])

    return E, X, Y
end

function projecterr(E, X, Y, Dinit::Dims, Dfinal::Dims)
    N = length(X)
    acc=E^2
    for i=1:N-1
        acc += norm(Y[i][Dinit[i+1]+1 : end, 1 : Dinit[i+1]])^2
        acc += norm(Y[i][Dinit[i+1]+1 : end, Dinit[i+1]+1 : end])^2
        acc -= norm(X[i][1 : Dfinal[i], 1 : Dfinal[i+1], :])^2
        acc += norm(Y[i][1 : Dfinal[i+1], 1 : Dfinal[i+1]])^2
    end
    i=N
    acc += norm(Y[i][Dinit[i+1]+1 : end, 1 : Dinit[i+1]])^2
    acc -= norm(X[i][1 : Dfinal[i], 1 : Dfinal[i+1], :])^2
    return sqrt(acc)
end
function projecterr(E, X, Y, Dinit::Dims, i::Int, Di::Int)
    Dfinal = (Dinit[1:i-1]..., Di, Dinit[i+1:end]...)
    projecterr(E, X, Y, Dinit, Dfinal)
end
function projecterr(E, X, Y, Dinit::Dims)
    projecterr(E, X, Y, Dinit, Dinit)
end

function expH2(E, X, Y, Dinit, f)
    N = length(X)
    d = [min(f, size(X[i])[3]) for i=1:N]
    acc=E
    for i=1:N-1
        acc += norm(Y[i][Dinit[i+1]+1 : d[i]*Dinit[i], 1 : Dinit[i+1]])
        acc += norm(Y[i][Dinit[i+1]+1 : d[i]*Dinit[i], Dinit[i+1]+1 : d[i+1]*Dinit[i+2]])
    end
    i=N
    acc += norm(Y[i][Dinit[i+1]+1 : d[i]*Dinit[i], 1 : Dinit[i+1]])
    return sqrt(acc)
end
function expH2(E, X, Y, Dinit)
    N = length(X)
    acc=E
    for i=1:N-1
        acc += norm(Y[i][Dinit[i+1]+1 : end, 1 : Dinit[i+1]])
        acc += norm(Y[i][Dinit[i+1]+1 : end, Dinit[i+1]+1 : end])
    end
    i=N
    acc += norm(Y[i][Dinit[i+1]+1 : end, 1 : Dinit[i+1]])
    return sqrt(acc)
end
