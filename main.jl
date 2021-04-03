include("include/MPSDynamics.jl")
include("include/fundamentals.jl")
using Jacobi

fdir = "./"

dt=0.25 # time step
tmax=60 # simulation time

"""
Physical parameters:
"""
ωc = 1 # Bath Spectral density cut-off frequency
ω0 = .25 # Sites coupling energy
α = 0.03 # Bath Spectral density coupling strength
s = 1 # Ohmic spectral density
beta = "inf" # Bath inverse temperature

R = 20 # Sites separation
c = 2 # Phonon speed

N = 2 # number of sites

Nm = 35 # number of chainmodes

issoft=false # is the cut-off of the spectral density a soft one or a hard one?

# g1 = 1.
# g2 = 1.
# w1 = 3.
# w2 = 4.
#
# e1 = (g1^2*w1+g2^2*w2)/(g1^2+g2^2)
# e2 = (g1^2*w2+g2^2*w1)/(g1^2+g2^2)
# t = g1*g2*(w1-w2)/(g1^2+g2^2)

#cp = [[e1,e2], [t]]
cp = chaincoeffs_ohmic(Nm, α, s, beta, ωc=ωc, soft=issoft) # chain parameters

"""
MPS and MPO parameters
"""
Chimax = 20 #Max bond dimension
dhilbert = 20 #Hilbert Space dimension of the chain sites

"""
Define the MPO:
"""
H(Dmax,d) = thibautmpo(N, Nm, d, J=ω0, chainparams=cp, s=s, α=α, ωc=ωc, R=R, c_phonon=c, beta=beta, issoft=issoft, compression=false)
"""
Define the MPS:
"""
## Single excitation state ##
statelist = [unitcol(1,dhilbert) for i in 1:2*Nm] # all the chain modes are in vaccuum state
for i in 2:N-1 # no excitation on the sites 3 to N
    pushfirst!(statelist,unitcol(1, 2))
end
statelist = pushfirst!(statelist,unitcol(2,2)) # the excitation is on the second site
pushfirst!(statelist,unitcol(1, 2)) #No excitation on the first site
A(Dmax,d) = productstatemps(physdims(H(Dmax,d)), Dmax, statelist=statelist)

## Superposition of states ##
function superpositionmps(Dmax, d)
    B = productstatemps(physdims(H(Dmax,d)), Dmax, statelist=statelist)
    temp = zeros(ComplexF64, 1, 2, 2)
    temp[:,:,1] = [1/sqrt(2) 0]
    temp[:,:,2] = [0 1/sqrt(2)]
    B[1] = temp
    temp = zeros(ComplexF64, 2, 4, 2)
    temp[:,:,1] = [0 0 0 0;1 0 0 0]
    temp[:,:,2] = [1 0 0 0;0 0 0 0]
    B[2] = temp

    return B
end


"""
Observables:
"""
obs = [
    ["n1", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 1)) ],
    ["n2", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 2)) ],
    #["n3", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 3)) ],
    ["tr(ϱ)", (psi,args...) -> real(normmps(psi))],
    ["bathOcc", (psi,args...) -> real(measure1siteoperator(psi, numb(dhilbert), (N+1,N+2*Nm)))],
    ["coherence12", (psi, args...) -> measure2siteoperator(psi, crea(2), anih(2), 1,2) ]
    #["coherence13", (psi, args...) -> abs(measure2siteoperator(psi, crea(2), anih(2), 1,3)) ],
    #["coherence23", (psi, args...) -> abs(measure2siteoperator(psi, crea(2), anih(2), 2,3)) ]
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
    ["ntot", (psi, args...) -> real(measure1siteoperator(psi, numb(2), 1)) + real(measure1siteoperator(psi, numb(2), 2)) ], #+ real(measure1siteoperator(psi, numb(2), 3)) ],
    ["tr(ϱ)", (psi, args...) -> real(normmps(psi))]
]


# Initial State of the system
#InitState(Dmax,d) = A(Dmax,d)
InitState(Dmax,d) = superpositionmps(Dmax, d)

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


