include("fundamentals.jl")

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

function displacedchainmps(A::Vector{Any}, N::Int, Nm::Int; γ=nothing, chainparams=[fill(1.0,Nm),fill(1.0,Nm-1), 1.0], s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false)
"""
For a displacement gamma of the bath modes, compute the corresponding displaced operator on the 2*Nm-long chain and apply it to a given mps A.
"""
    # if no displacement vector was given, we construct one with gamma_k = -g_k/omega_k * exp(-i*k*R)
    if γ==nothing
        γ = [-sqrt(spectraldensity(k, c_phonon))/(abs(k)*c_phonon)*exp(-im*k*R*ωc/c_phonon) for k=0.01:0.01:1]
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
    for n=1:2*Nm
        d1, d2, dhilbert = size(A[N+n]) # left bond dim, right bond dim and system physical dim
        χ = d1*d2
        #W = zeros(ComplexF64,dhilbert,dhilbert) # Argument of the displacement operator

        bd = crea(dhilbert) # chain creation operator
        b = anih(dhilbert) # chain anihilation operator

        ι = 0 + 0.0im # complex displacement amplitude
        L = length(γ)
        δk = 0.01 # spacing of the k-modes
        for k=1:L
            Unk = sqrt(spectraldensity(k/L, c_phonon))*polynomial(k/L,beta,n) # Matrix element of the unitary transformation from the bath to the chain
            ι += γ[k]*Unk*δk
        end
        push!(displ, abs(ι)^2)

        W = ι*bd - conj(ι)*b  # Argument of the displacement operator

        Ap = permutedims(reshape(A[N+n], χ, dhilbert), [2,1]) # reshape the MPS tensor as a (d, χ)-matrix

        As = permutedims(exp(W)*Ap, [2,1]) # matrix multiplication B_n = exp(W)*A_n

        push!(B, reshape(As, d1, d2, dhilbert))
    end

    return B
end
