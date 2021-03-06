using LinearAlgebra

function iszerovector(V)
    s = 0
    for i=1:length(V)
        if iszero(V[i])==false
            return false
        elseif iszero(V[i])==true
            s+=1
        end
    end
    if isequal(s,length(V))
        return true
    else
        return "Error"
    end
end

function isparallel(u,v)
    """
    Return 0 if the vectors are not parallel and the coefficient of proprtionality between them otherwise
     """
    cosθ = dot(u,v)/(norm(u)*norm(v))

    if cosθ == 1
        prefactor = norm(v)/norm(u)
    else
        prefactor = 0
    end

    return prefactor
end

function deparallelization(M)
    K = [] #set of kept collums
    d2 = size(M)[2]
    T = zeros(ComplexF64,d2,d2) #transfer matrix

    push!(K, M[:,1])
    T[1,1] = 1
    for j=2:d2
        s = 0
        L = length(K)
        if iszerovector(M[:,j])
            T[:,j] = zeros(d2)
        else
            for i=1:L
                prefactor = isparallel(K[i], M[:,j])
                if  prefactor != 0
                    T[i,j] = prefactor
                else
                    s+=1
                end
            end
            if s==L
                T[L+1, j] = 1
                push!(K, M[:,j])
            end
        end
    end

    if length(K)<d2
        T = T[1:length(K), :]
    end

    Mtilda = zeros(ComplexF64,length(K[1]), length(K))
    for i=1:length(K)
        Mtilda[:,i] = K[i]
    end

    return Mtilda, T
end

function mpocompression!(M)
    L = length(M)

    ##########################
    ## Left-to-right sweeps ##
    ##########################
    print("Left --> Right MPO Compression Sweep \n")
    d = size(M[1])
    Transfer = [Float64.(unitmat(d[1]))]
    for i=1:L
        print("Tensor #",i,"\n")
        W = M[i]
        dims = size(W)
        d1 = dims[1]
        d2 = dims[2]*dims[3]*dims[4]
        A = reshape(W,d1,d2) # reshape the tensor as a rectangular matrix

        A = Transfer[i]*A # matrix multiplication with the previous transfer matrix

        dp = size(A)[1]
        A = reshape(A, dp, dims[2], dims[3], dims[4])
        A = permutedims(A,[1,4,3,2])
        dims = size(A)
        d1 = dims[1]*dims[2]*dims[3]
        d2 = dims[4]
        A = reshape(A, d1, d2)

        Atilda, T = deparallelization(A) # A = Atilda*T
        dtilda = size(Atilda,2)
        Wtilda = reshape(Atilda, dims[1], dims[2], dims[3], dtilda) # reshape the compressed matrix as a compressed tensor
        Wtilda = permutedims(Wtilda, [1,4,3,2])

        M[i] = Wtilda
        push!(Transfer,T)
    end

    ##########################
    ## Right-to-left sweeps ##
    ##########################
    # print("Left <-- Right MPO Compression Sweep \n")
    # d = size(M[L])
    # Transfer = [unitmat(d[2])]
    # for i=L:-1:1
    #     print("Tensor #",i,"\n")
    #     W = M[i]
    #
    #     W = permutedims(W,[2,1,3,4])
    #     dims = size(W)
    #     d1 = dims[1]
    #     d2 = dims[2]*dims[3]*dims[4]
    #     Wp = reshape(W,d1,d2)
    #     Ws = Transfer[L+1-i]*Wp
    #     beta = size(Ws)[1]
    #     Wt = reshape(Ws,beta,dims[2],dims[3],dims[4])
    #     Wt = permutedims(Wt,[2,1,3,4])
    #
    #     Wt = permutedims(Wt,[4,2,3,1])
    #     dims = size(Wt)
    #     d1 = dims[1]*dims[2]*dims[3]
    #     d2 = dims[4]
    #     A = reshape(Wt,d1,d2)
    #     Atilda, T = deparallelization(A)
    #
    #     dtilda = size(Atilda)[2]
    #     Wtilda = reshape(Atilda, dims[1], dims[2], dims[3], dtilda)
    #     Wtilda = permutedims(Wtilda,[4,2,3,1])
    #
    #     M[i] = Wtilda
    #     push!(Transfer,T)
    #
    # end
end
