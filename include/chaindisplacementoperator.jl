include("fundamentals.jl")

using QuadGK
using Jacobi

function SDTOhmic(t, c_phonon)
"""
Effective Ohmic Spectral density at inverse temperature beta.
The argument t is the dimensionless variable ω/ωc.
"""
    if t==0
        return 2*α/beta
    elseif t>-1 && t<1
        return α*t*ωc/c_phonon*(1+coth(beta*t*ωc*0.5))
    elseif abs(t)==1
        return α*t*ωc/c_phonon*(1+coth(beta*t*ωc*0.5))
    else
        return 0
    end
end

function spectraldensity(t, c_phonon)
    if beta=="inf"
        return abs(t)<=1 ? 2*α*t*ωc/c_phonon : 0
    else
        return SDTOhmic(t, c_phonon)
    end
end

function polynomial(k,beta,n)
    if beta=="inf"
        if n <= Nm
            return jacobi(2*k-1,n-1, 0, s)*sqrt(2*(n-1) + s + 1)
        else
            return jacobi(2*k-1,n-Nm-1, 0, s)*sqrt(2*(n-Nm-1) + s + 1)
        end
    else
        if n <= Nm
            return polybeta(k, n, achain, bchain, [1.])
        else
            return polybeta(k, n-Nm, achain, bchain, [1.])
        end
    end
end

function displacedchainmps(A::Vector{Any}, N::Int, Nm::Int; γ=nothing, chainparams=[fill(1.0,Nm),fill(1.0,Nm-1), 1.0], s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false)
"""
For a displacement gamma of the bath modes, compute the corresponding displaced operator on the 2*Nm-long chain and apply it to a given mps A.
"""
    # if no displacement vector was given, we construct one with gamma_k = -g_k/omega_k * exp(-i*k*R)
    δk = 0.01 # spacing of the k-modes
    if γ==nothing
        γ = [-sqrt(spectraldensity(k, c_phonon))/(abs(k)*c_phonon)*exp(-im*k*R*ωc/c_phonon) for k=0.01:δk:1]
    else
        δk = 1/length(γ)
    end

    if beta != "inf" # For finite temperature we use the chain params for the polynomials of the unitary transformation
        achain = chainparams[1]./ωc
    	bchain = (chainparams[2].^2)./ωc
    end

    B = Any[] # displaced chain MPS
    displ = Any[] # modulus square of the displacement amplitude

    # The system part of the MPS should be unaffected by the displacement operator
    for i=1:N
        push!(B, A[i])
        push!(displ, 0.0)
    end

    # Displacement operator for the chain
    ι = zeros(ComplexF64,2*Nm) # complex displacement amplitude
    for n=1:2*Nm
        d1, d2, dhilbert = size(A[N+n]) # left bond dim, right bond dim and system physical dim
        χ = d1*d2
        #W = zeros(ComplexF64,dhilbert,dhilbert) # Argument of the displacement operator

        bd = crea(dhilbert) # chain creation operator
        b = anih(dhilbert) # chain anihilation operator

        #ι = 0 + 0.0im # complex displacement amplitude
        L = length(γ)

        # for k=1:L
        #     Unk = sqrt(spectraldensity(k/L, c_phonon))*polynomial(k/L,beta,n) # Matrix element of the unitary transformation from the bath to the chain
        #     ι += n<=Nm ? conj(γ[k])*Unk*δk : γ[k]*Unk*δk
        # end
        ι[n] = n<=Nm ? -2*α*sqrt(2*(n-1) + s + 1)*quadgk(k->jacobi(2*k-1,n-1,0,1)*exp(im*k*R*ωc/c_phonon),0,1)[1] : conj(ι[n-Nm])
        push!(displ, abs(ι[n])^2)

        W = ι[n]*bd - conj(ι[n])*b  # Argument of the displacement operator

        Ap = permutedims(reshape(A[N+n], χ, dhilbert), [2,1]) # reshape the MPS tensor as a (d, χ)-matrix

        As = permutedims(exp(W)*Ap, [2,1]) # matrix multiplication B_n = exp(W)*A_n

        push!(B, reshape(As, d1, d2, dhilbert))
    end

    #B = mpsrightnorm!(B)
    return B
end

function alternativedisplacedchainmps(A::Vector{Any}, N::Int, Nm::Int; γ=nothing, chainparams=[fill(1.0,Nm),fill(1.0,Nm-1), 1.0], s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false)
"""
For a displacement gamma of the bath modes, compute the corresponding displaced operator on the 2*Nm-long chain and apply it to a given mps A.
"""
    # if no displacement vector was given, we construct one with gamma_k = -g_k/omega_k * exp(-i*k*R)
    δk = 0.01 # spacing of the k-modes
    if γ==nothing
        γ = [-sqrt(spectraldensity(k, c_phonon))/(abs(k)*c_phonon)*exp(-im*k*R*ωc/c_phonon) for k=0.01:δk:1]
    else
        δk = 1/length(γ)
    end

    if beta != "inf" # For finite temperature we use the chain params for the polynomials of the unitary transformation
        achain = chainparams[1]./ωc
    	bchain = (chainparams[2].^2)./ωc
    end

    B = Any[] # displaced chain MPS
    displ = Any[] # modulus square of the displacement amplitude

    # The system part of the MPS should be unaffected by the displacement operator
    for i=1:N
        push!(B, A[i])
        push!(displ, 0.0)
    end

    # Displacement operator for the chain
    ι = zeros(ComplexF64,2*Nm) # complex displacement amplitude
    for n=1:2*Nm
        d1, d2, dhilbert = size(A[N+n]) # left bond dim, right bond dim and system physical dim

        # for k=1:L
        #     Unk = sqrt(spectraldensity(k/L, c_phonon))*polynomial(k/L,beta,n) # Matrix element of the unitary transformation from the bath to the chain
        #     ι += n<=Nm ? conj(γ[k])*Unk*δk : γ[k]*Unk*δk
        # end
        ι[n] = n<=Nm ? -2*α*sqrt(2*(n-1) + s + 1)*quadgk(k->jacobi(2*k-1,n-1,0,1)*exp(im*k*R*ωc/c_phonon),0,1)[1] : conj(ι[n-Nm])

        push!(B,A[N+n])

        for m=0:dhilbert-1
            B[end][1,1,m+1] = exp(-0.5*abs(ι[n])^2)*(ι[n])^m/sqrt(factorial(big(m)))
        end
    end

    return B
end

function directdisplacedchainmps(A::Vector{Any}, N::Int, Nm::Int; γ=nothing, chainparams=[fill(1.0,Nm),fill(1.0,Nm-1), 1.0], s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false)
"""
For a displacement gamma of the bath modes, compute the corresponding displaced operator on the 2*Nm-long chain and apply it to a given mps A.
"""
    coupling_stored = zeros(ComplexF64,N,Nm) # just need NxNm because the two chains have the same coupling coeff up to complex conjugation
    fnamecc = "chaincouplings_ohmic_R$(R)_a$(α)_wc$(ωc)_xc$(ωc/c)_beta$(beta).csv"
    lcouplings = Nm # number of stored coeff.
    chaincouplings = readdlm(fnamecc,',',ComplexF64,'\n')

    v = chaincouplings[2,1:Nm]

    M = Tridiagonal(chainparams[2], chainparams[1], chainparams[2])

    ι = M\v
    ι = vcat(-1*conj(ι), -1*ι)

    B = Any[] # displaced chain MPS
    displ = Any[] # modulus square of the displacement amplitude

    # The system part of the MPS should be unaffected by the displacement operator
    for i=1:N
        push!(B, A[i])
        push!(displ, 0.0)
    end

    # Displacement operator for the chain
    for n=1:2*Nm
        d1, d2, dhilbert = size(A[N+n]) # left bond dim, right bond dim and system physical dim
        χ = d1*d2
        #W = zeros(ComplexF64,dhilbert,dhilbert) # Argument of the displacement operator

        bd = crea(dhilbert) # chain creation operator
        b = anih(dhilbert) # chain anihilation operator

        W = ι[n]*bd - conj(ι[n])*b  # Argument of the displacement operator

        Ap = permutedims(reshape(A[N+n], χ, dhilbert), [2,1]) # reshape the MPS tensor as a (d, χ)-matrix

        As = permutedims(exp(W)*Ap, [2,1]) # matrix multiplication B_n = exp(W)*A_n

        push!(B, reshape(As, d1, d2, dhilbert))
    end

    return B
end
