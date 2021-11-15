include("fundamentals.jl")

function SDTOhmic(t)
"""
Effective Ohmic Spectral density at inverse temperature beta.
The argument t is the dimensionless variable ω/ωc.
"""
    if t==0
        return 2*α/beta
    elseif t>-1 && t<1
        return α*t*ωc*(1+coth(beta*t*ωc*0.5))
    elseif abs(t)==1
        return α*t*ωc*(1+coth(beta*t*ωc*0.5))
    else
        return 0
    end
end

function spectraldensity(k)
    if beta=="inf"
        return abs(k)<=1 ? 2*α*k*ωc/c_phonon : 0
    else
        return SDTOhmic(k)
    end
end

function polynomial(k,beta,n)
    if beta=="inf"
        if n <= Nm
            return jacobi(2*k-1,n-1, 0, s)
        else
            return jacobi(2*k-1,n-Nm-1, 0, s)
        end
    else
        if n <= Nm
            return polybeta(k, n, achain, bchain, [1.])
        else
            return polybeta(k, n-Nm, achain, bchain, [1.])
        end
    end
end

function displacedchainmps(A::Vector{Any}, γ::Array{Complex{Float64},1}, N::Int, Nm::Int, dhilbert::Int; J=0.2, chainparams, s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false)
"""
For a displacement gamma of the bath modes, compute the corresponding displaced operator on the 2*Nm-long chain and apply it to a given mps A.
"""
    if beta != "inf" # For finite temperature
        achain = chainparams[1]./ωc
    	bchain = (chainparams[2].^2)./ωc
    end

    D = Any[] # argument of the Displacement operator as a MPO
    B = Any[] # displaced chain MPS

    # The system part of the MPS should be unaffected by the displacement operator
    for i=1:N
        d1, d2, dsystem = size(A[i]) # left bond dim, right bond dim and system physical dim
        χ = d1*d2 # bond dimension of the MPO
        identity_system = zeros(ComplexF64,χ,χ,dsystem,dsystem)
        for j=1:d1
            for l=1:d2
                identity_system[j,l,:,:] = unitmat(dsystem)
            end
        end
        push!(D, identity_system)

        Ap = reshape(A[i], χ*dsystem) #reshape the MPS tensor as a vector
        M = reshape(identity_system, χ*dsystem, χ*dsystem) #reshape the MPO tensor as a square matrix

        As, info = exponentiate(M,1,Ap) # we apply the displacement operator to the MPS tensor: B = exp(M)*A
        if info.converged!=1
            error("exponentiation did not converge")
        end

        push!(B, reshape(As, d1, d2, dsystem))

    end

    # Displacement operator for the chain
    for n=1:2*Nm
        d1, d2, dhilbert = size(A[N+n])
        χ = d1*d2
        W = zeros(ComplexF64,χ,χ,dhilbert,dhilbert) # Argument of the displacement operator

        bd = crea(dhilbert) # chain creation operator
        b = anih(dhilbert) # chain anihilation operator

        ι = 0 #complex displacement amplitude
        L = length(γ)

        for k=1:L
            Unk = sqrt(spectraldensity(k/L))*polynomial(k/L,beta,n) # Matrix element of the unitary transformation from the chain to the bath
            ι += γ[k]*Unk
        end
        for j=1:χ
            for l=1:χ
                W[j,l,:,:] = ι*bd - conj(ι)*b
            end
        end
        push!(D, W)
        Ap = reshape(A[N+n], χ*dhilbert)
        M = reshape(W, χ*dhilbert, χ*dhilbert)

        As, info = exponentiate(M,1,Ap)
        if info.converged!=1
            error("exponentiation did not converge")
        end

        push!(B, reshape(As, d1, d2, dhilbert))
    end

    return B
end
