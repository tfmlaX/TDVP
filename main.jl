include("include/MPSDynamics.jl")
include("include/fundamentals.jl")
include("include/chaindisplacementoperator.jl")

using LaTeXStrings
using Jacobi

fdir = "./"

dt=1 # time step
tmax=100 # simulation time

"""
Physical parameters:
"""
ωc = 1 # Bath Spectral density cut-off frequency
ω0 = .15 # Sites coupling energy
E = [0.0, 0.0, 0.5] # system onsite energies. If unspecified, the energy will be set to zero
α = 0.4 # Bath Spectral density coupling strength
s = 1 # Ohmic spectral density
beta = "inf" # Bath inverse temperature

signature = [1, 1, -1] # signs of the system operators in the interaction Hamiltonian. If unspecified, the signature will be +1.
κ = 3 # multiplicative constant defining how much more coupled to the bath the Switch is

R = 5 # Sites separation
c = 1 # Phonon speed

N = 3 # The first one is the Switch, the second one is the bottom of the barrier and the last one is the top of the barrier
Nm = 25 # number of chainmodes

issoft=false # is the cut-off of the spectral density a soft one or a hard one?

cp = chaincoeffs_ohmic(Nm, α, s, beta, ωc=ωc, soft=issoft) # chain parameters

"""
MPS and MPO parameters
"""
Chimax = 20 #Max bond dimension
dhilbert = 10 #Hilbert Space dimension of the chain sites

"""
Define the MPO:
"""
H(Dmax,d) = correlatedenvironmentmpo(N, Nm, d, E=E, J=ω0, chainparams=cp, s=s, α=α, ωc=ωc, R=R, c_phonon=c, beta=beta, κ=κ, signature=signature, issoft=issoft, compression=false)
"""
Define the MPS:
"""
## Single excitation state ##
Nmodes = 2*Nm
statelist = [unitcol(1,dhilbert) for i in 1:Nmodes] # all the chain modes are in vaccuum state
for i in 2:N-1 # no excitation on the sites 3 to N
    pushfirst!(statelist,unitcol(1, 2))
end
statelist = pushfirst!(statelist,unitcol(2,2)) # the excitation is also on the second site
pushfirst!(statelist,unitcol(2, 2)) # excitation on the first site (but it's decoupled from the other sites)
localisedmps(Dmax,d) = productstatemps(physdims(H(Dmax,d)), Dmax, statelist=statelist)


"""
Observables:
"""
coupling_stored = zeros(ComplexF64,N,Nm) # just need NxNm because the two chains have the same coupling coeff up to complex conjugation
fnamecc = "chaincouplings_ohmic_R$(R)_a$(α)_wc$(ωc)_xc$(ωc/c)_beta$(beta).csv"
chaincouplings = []
try
    global chaincouplings = readdlm(fnamecc,',',ComplexF64,'\n')
catch error_storage
    if isa(error_storage, ArgumentError)
        _ = H(Chimax,dhilbert)
        global chaincouplings = readdlm(fnamecc,',',ComplexF64,'\n')
    end
end

lcouplings = Nm # number of stored coeff.

function wdisp1(d,n)
        cpl = chaincouplings[2,n]
        return cpl*anih(d) + conj(cpl)*crea(d)
end

function wdisp2(d,n)
        cpl = chaincouplings[2,n]
        return cpl*crea(d) + conj(cpl)*anih(d)
end

function energyshift(psi, dhilbert)
    S = 0
    for n=1:Nm
        S += measure2siteoperator(psi, numb(2), wdisp1(dhilbert,n), 2,N+n)
        S += measure2siteoperator(psi, numb(2), wdisp2(dhilbert,n), 2,N+Nm+n)
    end
    return real(S)
end

obs = [
    ["n1", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 1)) ],
    ["n2", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 2)) ],
    ["n3", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 3)) ],
    #["tr(ϱ)", (psi,args...) -> real(normmps(psi))],
    ["bathOcc", (psi,args...) -> real(measure1siteoperator(psi, numb(dhilbert), (N+1,N+Nmodes)))],
    ["energyshift", (psi, args...) -> real(energyshift(psi, dhilbert))],
    #["coherence12", (psi, args...) -> measure2siteoperator(psi, crea(2), anih(2), 1,2) ]
    #["coherence13", (psi, args...) -> abs(measure2siteoperator(psi, crea(2), anih(2), 1,3)) ]
    ["coherence23", (psi, args...) -> real(measure2siteoperator(psi, crea(2), anih(2), 2,3)) ]
    #["modesOcc", (psi,args...) -> nbath(psi)]
]

"""
Parameters to be stored:
"""
pars = [
    ["ωc",ωc],
    ["ω0",ω0],
    ["beta",beta],
    ["α",α],
    ["s",s],
    ["R",R],
    ["c",c],
    ["N",N],
    ["Nm",Nm],
    ["E",E],
    ["sign",signature],
    ["κ",κ],
    ["issoft",issoft],
    ["tmax",tmax]
]

"""
Convergence parameters:

Check for convergence in the local Hilbert space dimension d and the bond-dimension Dmax
"""
cpars = [
    ["Dmax",[Chimax]],
    ["d",[dhilbert]]
]

"""
Convergence observables:

Define the observables to check for convergence in
(note that these should return a single real value)
"""

cobs = [
    ["ntot", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 2)) + real(measure1siteoperator(psi, numb(2), 3)) ],
    ["tr(ϱ)", (psi, args...) -> real(normmps(psi))]
]


# Initial State of the system
#InitState(Dmax,d) = localisedmps(Dmax,d)
#InitState(Dmax,d) = superpositionmps(Dmax, d)
InitState(Dmax, d) = displacedchainmps(localisedmps(Dmax,d), N, Nm; s=s, α=α, ωc=ωc, R=R, c_phonon=c, beta =beta, issoft=false) # the initial state will be a displaced chain corresponding to a excitation sitting since -\infty at the bottom of the barrier. The reorganization energy -\lambda is induced by this initial state.

# Run the time evolution (dat is a dictionnary containing the time-series of the observables)
B, convdat, dat, convplts, deltas =
    convtdvp(dt, tmax, InitState, H, fdir;
             params=pars,
             observables=obs,
             convobs=cobs,
             convparams=cpars,
             lightcone=false,
             verbose=true,
             save=true,
             timed=false
             )

"""
Plots: convergence, populations in the site basis, populations in the eigen-basis
"""

# Show convergence plots:
#convplts["ntot"]
#convplts["tr(ϱ)"]

# Plot data:

# Plot sites populations
time = [i for i in 0:dt:tmax]
popplot = plot(ωc.*time, [1.0.-dat["n3"], dat["n3"]], label=["Site 2" "Site 3"], xlabel=L"\omega_c t", ylabel="Population", bg_legend=:transparent)
display(popplot)

shiftplot= plot(ωc.*time, dat["energyshift"], label=:none, xlabel =L"\omega_c t", ylabel=L"\Delta E")
display(shiftplot)

bathOcc = dat["bathOcc"]

Z = zeros(Nmodes,length(time))

for t=1:length(time)
    for n=1:Nm
        Z[n,t] = bathOcc[t][2*Nm+1 - n]
    end
    for n=1:Nm
        Z[Nm+n,t] = bathOcc[t][n]
    end
end

modes = vcat([i for i=-Nm+1:0],[i for i=0:Nm-1]);

spectroplot = heatmap(modes, ωc.*time, Z',guidefontsize=18,tickfontsize=12, title=L"Displaced ; Nm=$(Nm) ; d=$dhilbert ; \alpha=$α" )
xlabel!("Chain Mode")
ylabel!(L"\omega_c t")
display(spectroplot)
