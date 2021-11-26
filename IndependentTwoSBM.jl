using LinearAlgebra
using DifferentialEquations
using Jacobi
using QuadGK
using Plots
using Markdown

## Physical Parameters
α = 0.12 # Kondo param
s = 1 # Ohmicity
ωc = 1 # Bath cut-off frequency
c = 1 # Speed of sound
Nm = 100 # Number of chain modes

E1 = 0 # Energy of the first site; H_S = E1 σ1^z + E2 σ2^z
E2 = 0
delta = E1 - E2

#R = 10 # Sites separation

for R in [5,10,20,50]

    ## Construction of the system of ODEs
    # Coupling constant between site number x and mode number n
    function γ(n,x)
        polynomial(t) = jacobi(2*t-1,n-1, 0, s)*exp(-im*t*(x-1)*R*ωc/c)*t^s
        return sqrt(2*α*(2*(n-1) + s + 1))*ωc*quadgk(polynomial, 0, 1)[1]
    end

    # Onsite and hopping energies of chain modes
    ω = [(ωc/2)*(1 + (s^2)/((s+2n)*(2+s+2n))) for n in 0:(Nm-1)]
    t = [ωc*(1+n)*(1+s+n)/((s+2+2n)*(s+3+2n))*sqrt((3+s+2n)/(1+s+2n)) for n in 0:(Nm-2)]

    # The bath Hamiltonian is made of two tight-binding chain obtained after chain-mapping
    # The interaction Hamiltonian is H_int = ∑_i∑_n σi^(+) (γin b_n + γin^* d_n) + hc

    # The state has the for |ϕ> = A(t)|e,g,{0,0}> + B(t)|g,e,{0,0}> + ∑ cn(t)|g,g,{1_n, 0}> + ∑ c'n(t)|g,g,{0,1_n}>

    # Matrix defining the system of ODEs
    M = zeros(ComplexF64,2+2*Nm, 2+2*Nm)

    # Coefficients for dA/dt and dC0/dt
    M[1,1] = -im*delta
    γ01 = γ(1,1)
    M[1,3] = -im*γ01
    M[1,2+Nm+1] = -im*conj(γ01)

    # Coefficient for dB/dt and dCn/dt
    M[2,2] = im*delta
    for n=1:Nm
        γn2 = γ(n,2)
        M[2,2+n] = -im*γn2
        M[2,2+Nm+n] = -im*conj(γn2)
        M[2+n,2] = -im*conj(γn2)
        M[2+Nm+n,2] = -im*γn2
        M[2+n,2+n] = -im*ω[n]
        M[2+n,2+n+1] = n+1>Nm-1 ? continue : -im*t[n+1]
        M[2+n,2+n-1] = n-1<=0 ? continue : -im*t[n-1]
        M[2+Nm+n,2+Nm+n] = -im*ω[n]
        if n<Nm
            M[2+Nm+n,2+Nm+n+1] = n+1>Nm-1 ? continue : -im*t[n+1]
        end
        M[2+Nm+n,2+Nm+n-1] = n-1<=0 ? continue : -im*t[n-1]
    end

    M[2+1,1] = -im*conj(γ01)
    M[2+Nm+1, 1] = -im*γ01

    # We are going to solve the problem in its eigen-basis and transform back afterward
    F = eigen(M)
    D = Diagonal(F.values)
    V = F.vectors # Collection of eigen-vectors = Passage matrix

    ## Initial state

    ϕ0 = [0. for n=1:2*Nm] # cn(0) = 0 and c'n(0) = 0
    pushfirst!(ϕ0,0) # B(0) = 0
    pushfirst!(ϕ0,1) # A(0) = 1

    ψ0 = inv(V)*ϕ0 #Initial state in the eigen-problem basis

    ## Integration parameters
    tspan = (0.0,120.0)
    dt = 0.001 # timestep
    Δt = tspan[2] - tspan[1]
    time = [tspan[1]+i*dt for i=0:Int(Δt/dt)]

    ## Direct integration (because D does not depend on time) without ODEProblem ##
    # ψ = zeros(ComplexF64, 2*(Nm+1), length(time))
    # ψ[:,1] = ψ0
    # ϕ = zeros(ComplexF64, 2*(Nm+1), length(time))
    # ϕ[:,1] = ϕ0
    # for i=2:length(time)
    #     ψ[:,i] = exp(D.*dt)*ψ[:,i-1]
    #     normψ = sqrt(dot(ψ[:,i],ψ[:,i]))
    #     ψ[:,i] = ψ[:,i]./normψ
    #     ϕ[:,i] = V*ψ[:,i]
    #     normϕ = sqrt(dot(ϕ[:,i],ϕ[:,i]))
    #     ϕ[:,i] = ϕ[:,i]./normϕ
    # end

    ## Integration using ODEProblem (both methods give the same results) ##
    derivative!(dψ, ψ, p, t) = mul!(dψ, D, ψ)
    problem = ODEProblem(derivative!, ψ0, tspan)
    alg = RK4()
    sol = solve(problem, alg, dt=dt, verbose=:true)
    time = sol.t
    ϕ = zeros(ComplexF64, 2*(Nm+1), length(sol.t))
    P = zeros(Float64, 2*(Nm+1), length(time))
    Normalisation = zeros(length(time))
    for i=1:length(time)
        ϕ[:,i] = V*sol[i]
        normϕ = sqrt(dot(ϕ[:,i],ϕ[:,i]))
        ϕ[:,i] = ϕ[:,i]./normϕ
        P[:,i] = abs.(ϕ[:,i]).^2
        Normalisation[i] = sum(P[:,i])
    end

    ## Plotting the results

    siteplot = plot(time, [P[n,:] for n=1:2], label=["Site 1" "Site 2"], title="R = $(R)")
    xlabel!("t")
    ylabel!("P(t)")

    display(siteplot)

    modeplot = plot(time, P[3,:], label="Mode 1")
    for n=2:Nm
        plot!(time, P[2+n,:], label="Mode $(n)", title="R = $(R)")
    end
    xlabel!("t")
    ylabel!("Occupation")
    display(modeplot)

    # Chain spectrogram
    Z = zeros(2*Nm, length(time))
    for i=1:length(time)
        for n=1:Nm
            Z[n,i] = P[2+2*Nm+1 - n,i]
            Z[Nm+n,i] = P[2+n,i]
        end
    end
    modes = vcat([i for i=-Nm+1:0],[i for i=0:Nm-1])
    spectro = heatmap(modes, ωc.*time, Z',guidefontsize=18,tickfontsize=12, title="R = $(R)")
    xlabel!("Chain Mode")
    ylabel!("ωc t")
    display(spectro)

end
