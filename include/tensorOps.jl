using TensorOperations

#dir gives the open index where the parent is dir=1 (or first children if node is the head)
#Fs must be given in the correct order
#left = direction of parent
#right = direction of children
function updateleftenv(A, M, dir::Int)
    @tensor F[a,b,c] := conj(A[a,s'])*M[b,s',s]*A[c,s]
end
function updaterightenv(A, M)
    @tensor F[a,b,c] := conj(A[a,s'])*M[b,s',s]*A[c,s]
end
function updateleftenv(A, M, FL)
    @tensor F[a,b,c] := FL[a0,b0,c0]*conj(A[a0,a,s'])*M[b0,b,s',s]*A[c0,c,s]
end
function updateleftenv(A, M, dir::Int, F0)
    @tensor F[a,b,c] := F0[a0,b0,c0]*conj(A[a0,a,s'])*M[b0,b,s',s]*A[c0,c,s]
end
function updaterightenv(A, M, F1)
    @tensor F[a,b,c] := F1[a1,b1,c1]*conj(A[a,a1,s'])*M[b,b1,s',s]*A[c,c1,s]
end

function updateleftenv(A, M, dir::Int, F0, F1)# dir in {1,2}
    dir==1 && (Aperm = [1,3,2,4]; Mperm = [1,3,2,4,5])
    dir==2 && (Aperm = [1,2,3,4]; Mperm = [1,2,3,4,5])
    At = permutedims(A, Aperm)
    Mt = permutedims(M, Mperm)
    @tensoropt !(b,b0,b1) F[a,b,c] := F0[a0,b0,c0]*F1[a1,b1,c1]*conj(At[a0,a1,a,s'])*Mt[b0,b1,b,s',s]*At[c0,c1,c,s]
end
function updaterightenv(A, M, F1, F2)
    @tensoropt !(b,b1,b2) F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*conj(A[a,a1,a2,s'])*M[b,b1,b2,s',s]*A[c,c1,c2,s]
end
function updateleftenv(A, M, dir::Int, F0, F1, F2)# dir in {1,2,3}
    dir==1 && (Aperm = [1,3,4,2,5]; Mperm = [1,3,4,2,5,6])
    dir==2 && (Aperm = [1,2,4,3,5]; Mperm = [1,2,4,3,5,6])
    dir==3 && (Aperm = [1,2,3,4,5]; Mperm = [1,2,3,4,5,6])
    At = permutedims(A, Aperm)
    Mt = permutedims(M, Mperm)
    @tensoropt !(b,b0,b1,b2) F[a,b,c] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*conj(At[a0,a1,a2,a,s'])*Mt[b0,b1,b2,b,s',s]*At[c0,c1,c2,c,s]
end
function updaterightenv(A, M, F1, F2, F3)
    @tensoropt !(b,b1,b2,b3) F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a,a1,a2,a3,s'])*M[b,b1,b2,b3,s',s]*A[c,c1,c2,c3,s]
end

function applyH1(AC, M, F)
    @tensoropt !(b) HAC[a,s'] := F[a,b,c]*AC[c,s]*M[b,s',s]
end
function applyH1(AC, M, F0, F1)
    @tensoropt !(b0,b1) HAC[a0,a1,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*AC[c0,c1,s]*M[b0,b1,s',s]
end
function applyH1(AC, M, F0, F1, F2)
    @tensoropt !(b0,b1,b2) HAC[a0,a1,a2,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*AC[c0,c1,c2,s]*M[b0,b1,b2,s',s]
end
function applyH1(AC, M, F0, F1, F2, F3)
    @tensoropt !(b0,b1,b2,b3) HAC[a0,a1,a2,a3,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*AC[c0,c1,c2,c3,s]*M[b0,b1,b2,b3,s',s]
end
function applyH0(C, F0, F1)
    @tensor HC[α,β] := F0[α,a,α']*C[α',β']*F1[β,a,β']
end
