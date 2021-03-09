include("include/MPSDynamics.jl")
include("include/fundamentals.jl")
using Jacobi

fdir = "./"

dt=0.25 # time step
tmax=30 # simulation time

"""
Physical parameters:
"""
ω0 = .25 # Sites coupling energy
α = 0.01 # Bath Spectral density coupling strength
s = 1 # Ohmic spectral density
ωc = 1 # Bath Spectral density cut-off frequency
beta = 0.5 # Bath inverse temperature

c = 1 # Phonon speed

N = 1 # number of spin

Nm = 100 # number of chainmodes

issoft=false # is the cut-off of the spectral density a soft one or a hard one?

cp = chaincoeffs_ohmic(Nm, 0.25*α, s, beta, ωc=ωc, soft=issoft) # chain parameters

"""
MPS and MPO parameters
"""
Chimax = 20 #Max bond dimension
dhilbert = 20 #Hilbert Space dimension of the chain sites

"""
Define the MPO:
"""
H(Dmax,d) = spinbosonmpo(ω0, dhilbert, Nm, cp)
"""
Define the MPS:
"""
## Single excitation state ##
statelist = [unitcol(1,dhilbert) for i in 1:Nm] # all the chain modes are in vaccuum state
for i in 2:N-1 # no excitation on the sites 3 to N
    pushfirst!(statelist,unitcol(1, 2))
end
# statelist = pushfirst!(statelist,unitcol(1,2)) # the spin is in the up-z state
# statelist = pushfirst!(statelist,unitcol(2,2)) # the spin is in the down-z state
 statelist = pushfirst!(statelist,1/sqrt(2)*unitcol(1,2) .+ 1/sqrt(2)*unitcol(2,2)) # the spin is in a superposition of up-z and down-z
A(Dmax,d) = productstatemps(physdims(H(Dmax,d)), Dmax, statelist=statelist)


"""
Observables:
"""
obs = [
    ["sz", (psi, args...) -> real(measure1siteoperator(psi, sz, 1)) ],
    ["n1", (psi, args...) -> real(measure1siteoperator(psi, sm*sp, 1)) ],
    ["n2", (psi, args...) -> real(measure1siteoperator(psi, sp*sm, 1)) ],
    ["tr(ϱ)", (psi,args...) -> real(normmps(psi))],
    ["bathOcc", (psi,args...) -> real(measure1siteoperator(psi, numb(dhilbert), (N+1,N+Nm)))],
    ["coherence12", (psi, args...) -> measure1siteoperator(psi, sp, 1) ]
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
    ["Nm",Nm],
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
    ["tr(ϱ)", (psi, args...) -> real(normmps(psi))]
]


# Initial State of the system
InitState(Dmax,d) = A(Dmax,d)

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
#print("n1 + n2 + n3 = ",dat["n1"]+dat["n2"]+dat["n3"])
print("n1 + n2 = ",dat["n1"]+dat["n2"])
convplts["tr(ϱ)"]

# Plot data:

#scatter(dat["bathOcc"][end])
#xlabel!("Bath Modes")
#ylabel!("Population")

# Plot sites populations
time = [i for i in 0:dt:tmax]
#plot(time, [dat["n1"], dat["n2"], dat["n3"]], label=["Site 1" "Site 2" "Site 3"])
plot(time, [dat["n1"], dat["n2"]], label=["Site 1" "Site 2"])
xlabel!("Time")
ylabel!("Population")

# Plot eigen-states populations
pop2 = 0.5 .+ real.(dat["coherence12"])
pop1 = 1 .- pop2
plot(time, [pop1, pop2], label=["Lower State" "Upper State"])
xlabel!("Time")
ylabel!("Population")


