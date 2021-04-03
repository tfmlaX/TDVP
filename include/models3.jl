include("fundamentals.jl")
include("treeTDVP.jl")
include("deparallelization.jl")

using Jacobi
using QuadGK
using SpecialFunctions # for Euler Gamma function in the soft cut-off
using GSL

function thibautmpo(N::Int, Nm::Int, dhilbert::Int; J=0.2, chainparams, s=1, α=0.02, ωc=1, R=1, c_phonon=1, beta ="inf", issoft=false, compression=false)
    #=
    This function construct a MPO for a system made of N sites where a single excitation can propagate interacting with a bosonic bath with Nm modes.
    The interactions between the system and the bath are "long range", i.e. each site interacts with several modes.
    =#

    # Definition of the single excitation creation, anihilation operators on site x and the projector on site x
    creasite(x) = crea(2)
    anihsite(x) = anih(2)
    projsite(x) = [0 0;0 1]

    matlabdir = "./"
    datfname = "chaincoeffs_ohmic_a$(α)wc$(ωc)xc$(ωc/c_phonon)beta$(beta).csv"
    chaincoeffs = readdlm(string(matlabdir,datfname),',',Float64,'\n',skipstart=1)

    a_chain = chaincoeffs[:,1]
    b_chain = chaincoeffs[:,2]

    Norm = []
     
    function polybeta(t::Float64, n::Int, a::Array, b::Array, temp::Array)
    """
    polybeta recursively constructs the polynomials used to compute the coupling coefficients given the coefficients a and b
    this function is useful when working at finite temperature (beta != inf)	
    """
          if n==-1
              return 0
          elseif n==0
              if length(temp)>=2
                  temp[2] = 1
              else
                  push!(temp,1)
              end
              
              return 1
          elseif n==1
              pn = (t - a[n])
              if length(temp) == 2
                  push!(temp,pn)
              elseif length(temp) == 1
                  push!(temp, 1)
                  push!(temp,pn)
              end
              
              return pn
          else
              if length(temp)<n+1 && temp[1] == t
                  pn = (t - a[n])*polybeta(t,n-1,a,b,temp) - b[n-1]*polybeta(t,n-2,a,b,temp) #P_{n}(t) = (t-a_{n-1})P_{n-1} - b_{n-1}P_{n-2}
                  push!(temp, pn)
                  
                  return pn
              elseif length(temp)<n+1 && temp[1] != t
                  temp = [t]
                  pn = (t - a[n])*polybeta(t,n-1,a,b,temp) - b[n-1]*polybeta(t,n-2,a,b,temp) #P_{n}(t) = (t-a_{n-1})P_{n-1} - b_{n-1}P_{n-2}
                  push!(temp, pn)
                  
                  return pn
              elseif length(temp) == n+1 && temp[1] == t
                   pn = (t - a[n])*temp[n+1] - b[n-1]*temp[n]
                   push!(temp,pn)
                   
                   return pn
               elseif length(temp) == n+1 && temp[1] != t
                   temp = [t]
                   pn = (t - a[n])*polybeta(t,n-1,a,b,temp) - b[n-1]*polybeta(t,n-2,a,b,temp) #P_{n}(t) = (t-a_{n-1})P_{n-1} - b_{n-1}P_{n-2}
                   push!(temp, pn)
                   
                   return pn
               elseif length(temp) > n+1 && temp[1] == t
                   pn = temp[n+2]
                   
                   return pn
               else
                   temp = [t]
                   pn = (t - a[n])*polybeta(t,n-1,a,b,temp) - b[n-1]*polybeta(t,n-2,a,b,temp) #P_{n}(t) = (t-a_{n-1})P_{n-1} - b_{n-1}P_{n-2}
                   push!(temp, pn)
                   
                   return pn
               end
          end
      end


    # Definition of the coupling coefficient between the site x and the mode n for a Ohmic spectral density with a hard cut-off (Jacobi Polynomials) or a soft cut-off Laguerre Polynomials
    function γ(x::Int, n::Int, issoft::Bool; beta="inf", temp=[1.])
        if beta=="inf"
            if issoft==true
                polynomial0(t) = sf_laguerre_n(n,s,t)*exp(-im*t*x*R*ωc/c_phonon)*t^s*exp(-s)
                return sqrt(2*α*gamma(s + 1))*ωc*quadgk(polynomial0, 0, 1)[1]
            else
                polynomial(t) = 2*jacobi(2*t-1,n-1, 0, s)*exp(-im*t*(x-1)*R*ωc/c_phonon)*t^s
		return sqrt(2*α*(2*(n-1) + s + 1))*ωc*quadgk(polynomial, 0, 1)[1]

                # """ 2 discrete modes case """
                # g1 = 1.
                # g2 = 1.
                # w1 = 3.
                # w2 = 4.
                # if n==1
                #     return (g1^2*exp(-im*w1*(x-1)) + g2^2*exp(-im*w2*(x-1)))/sqrt(g1^2+g2^2)
                # else n==2
                #     return (g1*g2)*(exp(-im*w1*(x-1)) - exp(-im*w2*(x-1)))/sqrt(g1^2+g2^2)
                # end
            end
        elseif beta!="inf"
            polynomial(t) = polybeta(t,n-1,a_chain,b_chain,[t])
            integrand(t) = polynomial(t)*exp(im*t*(x-1)*R*ωc/c_phonon)*SDTOhmic(t)
            N2(t) = polynomial(t)^2*SDTOhmic(t)
            if length(Norm)<n
                push!(Norm,sqrt(quadgk(N2,-1,1)[1]))
            end
            return (ωc/Norm[n])*quadgk(integrand, -1, 1)[1]
            
        end
    end

    # Bath Ohmic Spectral Density for zero temperature chain mapping of the bath
    function SDOhmic(t)
        if t==0
            return 0
        elseif t>-1 && t<1
            return 2*α*abs(t)*ωc
        elseif abs(t)==1
            return 2*α*ωc
        else
            return 0
        end
    end

    # Bath Ohmic Spectral Density after the finite temperature chain mapping of the bath
    function SDTOhmic(t)
        if t==0
            return 4*α/beta
        elseif t>-1 && t<1
            return 2*α*t*ωc*(1+coth(beta*t*ωc*0.5))
        elseif abs(t)==1
            return 2*α*t*ωc*(1+coth(beta*t*ωc*0.5))
        else
            return 0
        end
    end

    # Construction of the MPO
    W = Any[] # list of the MPO's tensors
    d = 2 #N # Hilbert Space dimension of the sites operators
    u = unitmat(d)

    ### Construction of the system sites MPO ###
    print("Sites MPO \n")
    for x = 1:N-1
        print("Site ",x,"\n")
        cd = creasite(x)
        c = anihsite(x)
        P = projsite(x)

        D = 2*(x+2) # Bond dimension
        M = zeros(ComplexF64,D-2,D,d,d)
        M[1,1,:,:] = M[D-2,D,:,:] = u

        i = 2 # index counter
        M[1,i,:,:] = J*c
        M[1,i+1,:,:] = J*cd
        M[1,i+2*x,:,:] = M[1,i+1+2*x,:,:] = P
        # First site has a self-energy \omega_0, second sites has 0 energy
        # if x==1
        #     M[1,D,:,:] = J
        # end
        M[i,D,:,:] = cd
        M[i+1,D,:,:] = c
        i += 2

        while i<D-2
            M[i,i,:,:] = u
            i += 1
        end
        push!(W, M)
    end
    print("Last site before the chain \n")
    ### Last site before the bath chain doesn't have any coupling with the rest of the sites, so is size is Dx(D-2)xNxN
    cd = creasite(N)
    c = anihsite(N)
    P = projsite(N)

    D = 2*(N+1)
    M = zeros(ComplexF64,D, D, d, d)
    M[1,1,:,:] = M[D,D,:,:] = u
    i = 2 # index counter
    M[1,2*N,:,:] = M[1,1+2*N,:,:] = P
    #M[1,D,:,:] = 0.75*5*J*numb(2) # Last site has 0.75*\omega_0
    M[i,D,:,:] = cd
    M[i+1,D,:,:] = c

    while i<D-2
        M[i+2,i,:,:] = u
        i += 1
    end
    push!(W, M)

    print("Chain MPO \n")
    ### Construction of the bosonic bath MPO ###
    e = chainparams[1] # list of the energy of each modes
    t = chainparams[2] # list of the hopping energy between modes
    d = Int64(dhilbert) # Hilbert space dimension of the bosonic bath operators
    u = unitmat(d)
    # modes creation, anihilation and number operators
    bd = crea(d)
    b = anih(d)
    n = numb(d)

    if Nm != 1
        print("First Mode \n")
        ## First chain MPO
        D = 2*(N + 2) #Bond dimension
        M = zeros(ComplexF64,D-2, D, d, d)
        M[1,1,:,:] = M[D-2,D,:,:] = u
        M[1,D,:,:] = e[1]*n
        i = 2
        M[1,i,:,:] = t[1]*bd
        M[1,i+1,:,:] = t[1]*b
    #M[i,D,:,:] = b
    #M[i+1,D,:,:] = bd
    #i += 2

        a = 0 #site counter
        while i<D-2
            a+=1
            couplingcoeff = γ(a,1,issoft, beta=beta)
            M[i,D,:,:] = couplingcoeff*b
            M[i,i+2,:,:] = u
            i+=1
            M[i,D,:,:] = conj(couplingcoeff)*bd
            M[i,i+2,:,:] = u
            i+=1
        end
        M = reshape(M,D-2,D,d,d)
        push!(W, M)

        for m = 2:Nm-1
            print("Chain mode ",m,"\n")
            D = 2*(N + 2)
            M = zeros(ComplexF64,D, D, d, d)
            M[1,1,:,:] = M[D,D,:,:] = u
            M[1,D,:,:] = e[m]*n
            i = 2
            M[1,i,:,:] = t[m]*bd
            M[1,i+1,:,:] = t[m]*b
            M[i,D,:,:] = b
            M[i+1,D,:,:] = bd
            i += 2

            a = 0 #site counter
            while i<D
                a+=1
                couplingcoeff = γ(a, m, issoft,beta=beta)
                M[i,D,:,:] = couplingcoeff*b
                if i<D-1
                    M[i,i,:,:] = u
                end
                i+=1
                M[i,D,:,:] = conj(couplingcoeff)*bd
                if i<D
                    M[i,i,:,:] = u
                end
                i+=1
            end

            M = reshape(M,D,D,d,d)
            push!(W, M)
        end

	print("Last Mode of the First Chain \n")
        D = 2*(N + 2)
        M = zeros(ComplexF64,D, D, d, d)
        M[1,1,:,:] = M[D,D,:,:] = u
        M[1,D,:,:] = e[Nm]*n
        i = 2
        M[i,D,:,:] = b
        M[i+1,D,:,:] = bd
        i += 2

        a = 0 #site counter
        while i<D
            a+=1
            couplingcoeff = γ(a, Nm, issoft,beta=beta)
            M[i,D,:,:] = couplingcoeff*b
            if i<D-1
                M[i,i,:,:] = u
            end
            i+=1
            M[i,D,:,:] = conj(couplingcoeff)*bd
            if i<D
                M[i,i,:,:] = u
            end
            i+=1
        end

            M = reshape(M,D,D,d,d)
            push!(W, M)

	print("Second chain \n")
	for m = 1:Nm-1
            print("Chain mode ",m,"\n")
            D = 2*(N + 2)
            M = zeros(ComplexF64,D, D, d, d)
            M[1,1,:,:] = M[D,D,:,:] = u
            M[1,D,:,:] = e[m]*n
            i = 2
            M[1,i,:,:] = t[m]*bd
            M[1,i+1,:,:] = t[m]*b
            M[i,D,:,:] = b
            M[i+1,D,:,:] = bd
            i += 2

            a = 0 #site counter
            while i<D
                a+=1
                couplingcoeff = γ(a, m, issoft,beta=beta)
                M[i,D,:,:] = couplingcoeff*bd
                if i<D-1
                    M[i,i,:,:] = u
                end
                i+=1
                M[i,D,:,:] = conj(couplingcoeff)*b
                if i<D
                    M[i,i,:,:] = u
                end
                i+=1
            end

            M = reshape(M,D,D,d,d)
            push!(W, M)
        end

        print("Last mode \n")
        WNm = zeros(ComplexF64,D, 1, d, d)
        WNm[1,1,:,:] = e[Nm]*n
        WNm[2,1,:,:] = b
        WNm[3,1,:,:] = bd
        a = 0 #site counter
        i = 4 #row index
        while i<D
            a+=1
            couplingcoeff = γ(a,Nm, issoft, beta=beta)
            WNm[i,1,:,:] = couplingcoeff*bd
            i+=1
            WNm[i,1,:,:] = conj(couplingcoeff)*b
            i+=1
        end
        WNm[D,1,:,:] = u
        WNm = reshape(WNm,D,1,d,d)

    elseif Nm==1

        print("First and Last mode \n")
        D = 2*(N + 1) #Bond dimension
        WNm = zeros(ComplexF64,D, 1, d, d)
        WNm[1,1,:,:] = e[Nm]*n
        a = 0 #site counter
        i = 2 #row index
        while i<D
            a+=1
            couplingcoeff = γ(a,Nm, issoft, beta=beta)
            WNm[i,1,:,:] = couplingcoeff*b
            i+=1
            WNm[i,1,:,:] = conj(couplingcoeff)*bd
            i+=1
        end
        WNm[D,1,:,:] = u
        WNm = reshape(WNm,D,1,d,d)
    end

    W1 = W[1]
    if compression
        R = Any[W1[1:1,:,:,:], W[2:(N+Nm-1)]..., WNm]
        mpocompression!(R)
        return R
    else
        return Any[W1[1:1,:,:,:], W[2:(N+2*Nm-1)]..., WNm]
    end
end


"""
function chaincoeffs_ohmic(nummodes, α, s, beta="inf"; wc=1, soft=false)

Generates chain coefficients for an Harmonic bath coupled to a spin-1/2 with spectral density given by:

soft cutoff: J(ω) = 2παω_c^(1-s)ω^s * exp(-ω/ω_c)
hard cutoff: J(ω) = 2παω_c^(1-s)ω^s * θ(ω-ω_c)

The Hamiltonian is given by:

    H = (ω0/2)σz + Δσx + σxΣₖgₖ(bₖ†+bₖ) + Σₖωₖbₖ†bₖ

And the spectral density is defined by:

J(ω) ≡ πΣₖ|gₖ|²δ(ω-ωₖ)
"""

function chaincoeffs_ohmic(nummodes, α, s, beta="inf"; ωc=1, soft=false)
        if beta=="inf"
        if soft
            c0 = ωc*sqrt(2*α*gamma(s+1))
            e = [ωc*(2n + 1 + s) for n in 0:(nummodes-1)]
            t = [ωc*sqrt((n + 1)*(n + s + 1)) for n in 0:(nummodes-2)]
            return e, t, c0
        else
	    # For frequency modes, the polynomials are Jacobi polynomials
            c0 = sqrt(2α/(s+1))*ωc
            e = [(ωc/2)*(1 + (s^2)/((s+2n)*(2+s+2n))) for n in 0:(nummodes-1)]
            t = [ωc*(1+n)*(1+s+n)/((s+2+2n)*(s+3+2n))*sqrt((3+s+2n)/(1+s+2n)) for n in 0:(nummodes-2)]
            return e, t, c0

	    # For wave-vector modes, we have to construct the polynomials
	    #return getchaincoeffs(nummodes, α, s, beta, ωc)
        end
    else
        if soft
            throw(ErrorException("no data for soft cut-off at finite beta"))
        end
        return getchaincoeffs(nummodes, α, s, beta, ωc)
    end
end


function getchaincoeffs(nummodes, α, s, beta, ωc=1)
    matlabdir = "./"
    #astr = paramstring(α, 2)
    #bstr = paramstring(beta, 3)
    datfname = "chaincoeffs_ohmic_a$(α)wc$(ωc)xc$(ωc/c)beta$(beta).csv"
    chaincoeffs = readdlm(string(matlabdir,datfname),',',Float64,'\n',skipstart=1)
    es = ωc .* chaincoeffs[1:end,1]
    ts = ωc .* chaincoeffs[1:end-1,2]
    c0 = ωc * sqrt(chaincoeffs[end,2]/pi)
    Nmax = length(es)
    if Nmax < nummodes
        throw(ErrorException("no data for nummodes > $Nmax"))
    end
    return [es[1:nummodes], ts[1:nummodes-1], c0]
end

function spinbosonmpo(ω0, d, Nm, chainparams)
    print("Spinboson MPO called\n")
    u = unitmat(2)

    up(H, h, hd) = permutedims(cat(H, h, hd, u; dims=3), [3,1,2])

    c0 = chainparams[3]

    Hs = ω0*sx

    M=zeros(1,4,2,2)
    M[1, :, :, :] = up(Hs, c0*sz, c0*sz)

    chain = bathchain(Nm, d, chainparams)
    chain = chain.sites
    chain[end] = reshape(chain[end],4,1,d,d)
    print("Exit Spinboson MPO\n")
    return Any[M, chain...]
end

function bathchain(N::Int, d::Int, chainparams)
    b = anih(d)
    bd = crea(d)
    n = numb(d)
    u = unitmat(d)
    e = chainparams[1]
    t = chainparams[2]

    up(H, h, hd) = permutedims(cat(H, h, hd, u; dims=3), [3,1,2])
    down(H, h, hd) = permutedims(cat(u, hd, h, H; dims=3), [3,1,2])

    H=Vector{AbstractArray}()
    for i=1:N-1
        M=zeros(4, 4, d, d)
        M[4, :, :, :] = up(e[i]*n, t[i]*b, t[i]*bd)
        M[:, 1, :, :] = down(e[i]*n, b, bd)
        push!(H, M)
    end
    M=zeros(4, d, d)
    M[:, :, :] = down(e[N]*n, b, bd)
    push!(H, M)
    TreeNetwork(H)
end
