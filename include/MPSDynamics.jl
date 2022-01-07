using JLD, Random, Dates, Plots
include("config.jl")
include("chainTDVP.jl")
include("treeTDVP.jl")
include("models.jl")

function runtdvp!(dt, T, B, H, dir::String; params=[], observables=[], saveevery=nothing, verbose=false, save=false, timed=false, unid=nothing, log=true, lightcone=false)

    if !(save)
        return runtdvp!(dt, T, B, H, observables=observables, verbose=verbose, timed=timed, lightcone=lightcone)
    end

    if lightcone
        lc = LightCone(B)
    else
        lc = nothing
    end

    if dir[end] != '/'
        dir = string(dir,"/")
    end

    mkpath(dir)

    numobs = length(observables)
    numparams = length(params)
    if unid==nothing
        unid = randstring(5)
    end
    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))

    keylist = [observables[i][1] for i=1:numobs]

    if saveevery == nothing
        saveevery = numsteps*2 # will save only initial and final mps
    end

    try
        f = open(string(dir,"info.txt"))
        close(f)
    catch
        touch(string(dir,"info.txt"))
    end

    if log
        endpos = open(string(dir,"info.txt"), append=true) do f
            write(f,"\n\n")
            println("start time : $(now())")
            write(f,"start time : $(now()) \n")
            println("unid : $unid")
            write(f,"unid : $unid \n")
            for i in 1:numparams
                println(string(params[i][1], " : ",params[i][2]))
                write(f, string(params[i][1], " : ",params[i][2], " \n"))
            end
            print("observables : ")
            write(f,"observables : ")
            for i in 1:numobs
                print(keylist[i])
                print(", ")
                write(f, string(keylist[i], ", "))
            end
            print("\n")
            write(f, "\n")

            println()
            write(f,"\n >>>>>>>>>>>>>>>>>>>>")
            endpos = position(f)
            write(f,"<<<<<<<<<<<<<<<<<<<<                                                        ")
            return endpos
        end
    end

    res = Array{Any}(undef, numobs, numsteps+1)
    times = Vector{Any}()
    savesteps = Vector{Any}()

    tstart=now()
    jldopen(string(dir,"dat_",unid,".jld"), "w") do file
        for i in 1:length(params)
            write(file, params[i][1], params[i][2])
        end
        write(file, "obs", keylist)

        if lightcone
            println("time = 0, LCE = $(lc.edge)")
        else
            println("time = 0")
        end

        for tstep in 1:numsteps
            for i in 1:numobs
                res[i,tstep] = observables[i][2](B)
            end
            if tstep%saveevery == 1
                write(file, "tmps$tstep", B)
                push!(savesteps, tstep)
            end
            push!(times, t)

            B, F = timed ? tdvp1sweep_timed!(dt, B, H, F, lc, verbose=verbose) : tdvp1sweep!(dt, B, H, F, lc, verbose=verbose)
            t += dt
            if lightcone
                println("time = $(round(t;digits=2)), LCE = $(lc.edge)")
            else
                println("time = $(round(t;digits=2))")
            end
        end

        for i in 1:numobs
            res[i,numsteps+1] = observables[i][2](B)
        end
        push!(times, t)

        for i in 1:numobs
            write(file, observables[i][1], res[i,:])
        end
        write(file,"mpsfinal", B)
        write(file,"times",times)
        write(file,"savesteps",savesteps)
    end
    if log
        open(string(dir,"info.txt"), "a+") do f
            seek(f, endpos)
            write(f,"run completed at $(now())<<<<<<<<<<<<<<<<<<<<\n")
            write(f,string("run time : ",now()-tstart,"\n"))
            println(string("run time : ",now()-tstart))
        end
    end

    return B, Dict([(observables[i][1],res[i,:]) for i in 1:numobs]), unid
end
runtdvp_timed!(dt, T, B, H, dir::String; params=[], observables=[], saveevery=nothing, verbose=false, save=false, timed=false, unid=nothing, log=true, lightcone=false) = @time runtdvp!(dt, T, B, H, dir; params=params, observables=observables, saveevery=saveevery, verbose=verbose, save=save, timed=timed, unid=unid, log=log, lightcone=lightcone)

function runtdvp!(dt, T, B, H; observables=[], verbose=false, timed=false, lightcone=false)

    if lightcone
        lc = LightCone(B)
    else
        lc = nothing
    end

    numobs = length(observables)
    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))
    res = Array{Any}(undef, numobs, numsteps+1)

    if lightcone
        println("time = 0, LCE = $(lc.edge)")
    else
        println("time = 0")
    end

    for tstep in 1:numsteps
        for i in 1:numobs
            res[i,tstep] = observables[i][2](B)
        end

        B, F = timed ? tdvp1sweep_timed!(dt, B, H, F, lc; verbose=verbose) : tdvp1sweep!(dt, B, H, F, lc; verbose=verbose)
        t += dt
        if lightcone
            println("time = $(round(t;digits=2)), LCE = $(lc.edge)")
        else
            println("time = $(round(t;digits=2))")
        end
    end

    for i in 1:numobs
        res[i,numsteps+1] = observables[i][2](B)
    end
    return B, Dict([(observables[i][1],res[i,:]) for i in 1:numobs]), nothing
end
runtdvp_timed!(dt, T, B, H; observables=[], verbose=false, timed=false, lightcone=false) = @time runtdvp!(dt, T, B, H; observables=observables, verbose=verbose, timed=timed, lightcone=lightcone)

function convtdvp_multithread(dt, T, B::Function, H::Function, dir::String; thresh=DEFCONVTHRESH, params=[], convparams=[], observables=[], convob=nothing, saveevery=nothing, save=false, unid=nothing, saveplot=true, lightcone=false)

    verbose=false
    timed=false
    if !(save)
        return convtdvp_mulithread(dt, T, B, H; convparams=convparams, observables=observables, convob=convob, thresh=thresh, lightcone=lightcone)
    end

    if dir[end] != '/'
        dir = string(dir,"/")
    end

    mkpath(dir)

    if convob==nothing
        throw(error("convob must be specified"))
    end

    numobs = length(observables)
    numparams = length(params)
    numconvparams = length(convparams)
    if unid==nothing
        unid = randstring(5)
    end
    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))

    keylist = [observables[i][1] for i=1:numobs]

    if saveevery == nothing
        saveevery = numsteps*2 # will save only initial and final mps
    end

    try
        f = open(string(dir,"info.txt"))
        close(f)
    catch
        touch(string(dir,"info.txt"))
    end

    endpos = open(string(dir,"info.txt"), append=true) do f
        write(f,"\n\n")
        println("start time : $(now())")
        write(f,"start time : $(now()) \n")
        println("num threads : $(Threads.nthreads())")
        write(f,"num threads : $(Threads.nthreads()) \n")
        println("unid : $unid")
        write(f,"unid : $unid \n")
        for i in 1:numparams
            println(string(params[i][1], " : ",params[i][2]))
            write(f, string(params[i][1], " : ",params[i][2], "\n"))
        end
        for i in 1:numconvparams
            println(string(convparams[i][1], " : ",convparams[i][2]))
            write(f, string(convparams[i][1], " : ",convparams[i][2], "\n"))
        end
        write(f, string("convob : ", convob[1], "\n"))
        println(string("convob : ", convob[1]))

        print("observables : ")
        write(f,"observables : ")
        for i in 1:numobs
            print(keylist[i])
            print(", ")
            write(f, string(keylist[i], ", "))
        end
        print("\n")
        write(f, "\n")

        println()
        write(f,"\n >>>>>>>>>>>>>>>>>>>>")
        endpos = position(f)
        write(f,"<<<<<<<<<<<<<<<<<<<<                                                        ")
        for i in 1:numconvparams+2
            write(f,"\n                                                                      ")
        end
        return endpos
    end

    runparams = [ntuple(x->convparams[x][2][end], numconvparams)]
    for i in 1:numconvparams
        numvals = length(convparams[i][2])
        for j in numvals-1:-1:1
            vals = ntuple(x->convparams[x][2][x!=i ? end : j], numconvparams)
            push!(runparams, vals)
        end
    end
    numruns = length(runparams)
    res=Vector{Any}(undef, numruns)

    tstart=now()
    Threads.@threads for i in 1:numruns # note that multithreading in Julia is still experimental
        global Bret, datret
        vals = runparams[i]
        if i==1
            Bret, datret = runtdvp!(dt, T, B(vals...), H(vals...), dir, params=params, observables=observables, verbose=false, timed=timed, saveevery=saveevery, save=true, unid=unid, log=false, printtime=false, lightcone=lightcone)
            res[i] = datret[convob[1]]
        else
            dat=runtdvp!(dt, T, B(vals...), H(vals...), observables=[[convob[1], psi->convob[2](psi, vals...)]], verbose=false, timed=timed, printtime=false, lightcone=lightcone)[2]
            res[i] = dat[convob[1]]
        end
    end

    convdat=Vector{Any}(undef, numconvparams)
    finalrun = popfirst!(res)
    for i in 1:numconvparams
        tmp=zeros(length(res[1]), 0)
        tmp=hcat(tmp, finalrun)
        numvals = length(convparams[i][2])
        for j in 1:numvals-1
            tmp=hcat(tmp, popfirst!(res))
        end
        convdat[i]=tmp
    end

    deltas = [[rmsd(convdat[i][:,1], convdat[i][:,j]) for j in length(convparams[i][2]):-1:2] for i in 1:numconvparams]

    jldopen(string(dir,"convdat_",unid,".jld"), "w") do file
        write(file, "convdat", convdat)
        write(file, "deltas", deltas)
        write(file, "convparams", convparams)
    end

    open(string(dir,"info.txt"), "a+") do f
        seek(f, endpos)
        write(f,"run completed at $(now())<<<<<<<<<<<<<<<<<<<<\n")
        write(f,string("run time : ",now()-tstart,"\n"))
        println(string("run time : ",now()-tstart))

        for i in 1:numconvparams
            ilowconval = findfirst(x -> x < thresh, deltas[i])
            if ilowconval != nothing
                lowconval = convparams[i][2][ilowconval]
                write(f, string("converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh,"\n"))
                println(string("converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh,"\n"))
            else
                write(f, string("not converged in ",convparams[i][1]," with threshold of ",thresh,"\n"))
                println(string("not converged in ",convparams[i][1]," with threshold of ",thresh,"\n"))
            end
        end
    end

default(size = (800,600), reuse = true, title=unid, legend = true)
subplts=[]
subplts = [plot([i*dt for i=0:numsteps], convdat[i], label=reshape([string(convparams[i][1]," = ", convparams[i][2][j]) for j in length(convparams[i][2]):-1:1], 1, length(convparams[i][2]))) for i in 1:numconvparams]
plt = plot(subplts..., layout=(numconvparams,1), xlabel="t", ylabel=convob[1])
if saveplot
    savefig(plt, string(dir,"convplot_",unid,".pdf"))
end

return Bret, convdat, datret, plt, deltas, unid
end

function convtdvp_multithread(dt, T, B::Function, H::Function; thresh=DEFCONVTHRESH, convparams=[], observables=[], convob=nothing, lightcone=false)

    verbose=false
    timed=false
    if convob==nothing
        throw(error("convob must be specified"))
    end

    numobs = length(observables)
    numconvparams = length(convparams)

    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))

    runparams = [ntuple(x->convparams[x][2][end], numconvparams)]
    for i in 1:numconvparams
        numvals = length(convparams[i][2])
        for j in numvals-1:-1:1
            vals = ntuple(x->convparams[x][2][x!=i ? end : j], numconvparams)
            push!(runparams, vals)
        end
    end
    numruns = length(runparams)
    res=Vector{Any}(undef, numruns)

    tstart=now()
    Threads.@threads for i in 1:numruns  # note that multithreading in Julia is still experimental
        vals = runparams[i]
        if i==1
            global Bret, datret
            Bret, datret = runtdvp!(dt, T, B(vals...), H(vals...), observables=observables, verbose=false, timed=false, printtime=false, lightcone=lightcone)
            res[i] = datret[convob[1]]
        else
            dat=runtdvp!(dt, T, B(vals...), H(vals...), observables=[[convob[1], psi->convob[2](psi, vals...)]], verbose=false, timed=false, printtime=false, lightcone=lightcone)[2]
            res[i] = dat[convob[1]]
        end
    end

    convdat=Vector{Any}(undef, numconvparams)
    finalrun = popfirst!(res)
    for i in 1:numconvparams
        tmp=zeros(length(res[1]), 0)
        tmp=hcat(tmp, finalrun)
        numvals = length(convparams[i][2])
        for j in numvals-1:-1:1
            tmp=hcat(tmp, popfirst!(res))
        end
        convdat[i]=tmp
    end

    default(size = (800,600), reuse = true, legend = true)
    subplts=[]
    subplts = [plot([i*dt for i=0:numsteps], convdat[i], label=reshape([string(convparams[i][1]," = ", convparams[i][2][j]) for j in length(convparams[i][2]):-1:1], 1, length(convparams[i][2]))) for i in 1:numconvparams]
    plt = plot(subplts..., layout=(numconvparams,1), xlabel="t", ylabel=convob[1])

    deltas = [[rmsd(convdat[i][:,1], convdat[i][:,j]) for j in length(convparams[i][2]):-1:2] for i in 1:numconvparams]

    println(string("run time : ",now()-tstart))
    for i in 1:numconvparams
        ilowconval = findfirst(x -> x < thresh, deltas[i])
        if ilowconval != nothing
            lowconval = convparams[i][2][ilowconval]
            println(string("converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh,"\n"))
        else
            println(string("not converged in ",convparams[i][1]," with threshold of ",thresh,"\n"))
        end
    end

    return Bret, convdat, datret, plt, deltas, nothing
end

function convtdvp(dt, T, B::Function, H::Function, dir::String; thresh=DEFCONVTHRESH, params=[], convparams=[], observables=[], convobs=nothing, saveevery=nothing, verbose=false, save=false, timed=false, unid=nothing, saveplot=true, lightcone=false, multithread=false)

    if !(save)
        return convtdvp(dt, T, B, H; convparams=convparams, observables=observables, convobs=convobs, verbose=verbose, timed=timed, multithread=multithread, lightcone=lightcone)
    elseif multithread
        return convtdvp_multithread(dt, T, B, H, dir; thresh=thresh, params=params, convparams=convparams, observables=observables, convobs=convobs, saveevery=saveevery, save=save, unid=unid, saveplot=saveplot, lightcone=lightcone)
    end

    if dir[end] != '/'
        dir = string(dir,"/")
    end

    mkpath(dir)

    if convobs==nothing
        throw(error("convobs must be specified"))
    end

    numobs = length(observables)
    numparams = length(params)
    numconvparams = length(convparams)
    numconvobs = length(convobs)
    if unid==nothing
        unid = randstring(5)
    end
    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))
    obkeys = [observables[i][1] for i=1:numobs]
    convobkeys = [convobs[i][1] for i=1:numconvobs]

    for i in 1:numconvobs
        if !(in(convobkeys[i], obkeys))
            push!(observables, convobs[i])
        end
    end
    numobs = length(observables)
    obkeys = [observables[i][1] for i=1:numobs]

    if saveevery == nothing
        saveevery = numsteps*2 # will save only initial and final mps
    end
    try
        f = open(string(dir,"info.txt"))
        close(f)
    catch
        touch(string(dir,"info.txt"))
    end
    endpos = open(string(dir,"info.txt"), append=true) do f
        write(f,"\n\n")
        println("start time : $(now())")
        write(f,"start time : $(now()) \n")
        println("unid : $unid")
        write(f,"unid : $unid \n")
        for i in 1:numparams
            println(string(params[i][1], " : ",params[i][2]))
            write(f, string(params[i][1], " : ",params[i][2], "\n"))
        end
        for i in 1:numconvparams
            println(string(convparams[i][1], " : ",convparams[i][2]))
            write(f, string(convparams[i][1], " : ",convparams[i][2], "\n"))
        end
        print("convergence observables : ")
        write(f,"convergence observables : ")
        for i in 1:numconvobs
            print(convobkeys[i])
            print(", ")
            write(f, string(convobkeys[i], ", "))
        end
        print("\n")
        write(f, "\n")
        print("observables : ")
        write(f,"observables : ")
        for i in 1:numobs
            print(obkeys[i])
            print(", ")
            write(f, string(obkeys[i], ", "))
        end
        print("\n")
        write(f, "\n")
        println()
        write(f,"\n >>>>>>>>>>>>>>>>>>>>")
        endpos = position(f)
        write(f,"<<<<<<<<<<<<<<<<<<<<                                                        ")
        for i in 1:numconvparams*numconvobs + 2
            write(f,"\n                                                                      ")
        end
        return endpos
    end

    convdat=fill(zeros(numsteps+1, 0), numconvobs, numconvparams)

    tstart=now()
    for i in 1:numconvparams
        numvals = length(convparams[i][2])
        vals = [convparams[i][2][end] for i=1:numconvparams]
        for j in numvals-1:-1:1
            vals[i] = convparams[i][2][j]
            for k in 1:numconvparams
                print(convparams[k][1]," : ",vals[k]," ")
            end
            println()
            dat=runtdvp!(dt, T, B(vals...), H(vals...), observables=[[convobs[i][1], psi->convobs[i][2](psi, vals...)] for i in 1:numconvobs], verbose=verbose, timed=timed, lightcone=lightcone)[2]
            for k=1:numconvobs
                convdat[k,i] = hcat(convdat[k,i], dat[convobs[k][1]])
            end
        end
    end

    vals = [convparams[i][2][end] for i=1:numconvparams]
    for k in 1:numconvparams
        print(convparams[k][1]," : ",vals[k]," ")
    end
    println()
    Bret, dat = runtdvp!(dt, T, B(vals...), H(vals...), dir, params=params, observables=observables, verbose=verbose, timed=timed, saveevery=saveevery, save=true, unid=unid, log=false, lightcone=lightcone)

    for i in 1:numconvparams
        for j in 1:numconvobs
            convdat[j,i] = hcat(dat[convobs[j][1]], convdat[j,i])
        end
    end

    deltas = [[[rmsd(convdat[k,i][:,1], convdat[k,i][:,j]) for j in length(convparams[i][2]):-1:2] for i in 1:numconvparams] for k in 1:numconvobs]
    jldopen(string(dir,"convdat_",unid,".jld"), "w") do file
        for i in 1:numconvobs
            write(file, convobs[i][1], convdat[i,:])
            write(file, string("deltas_",convobs[i][1]), deltas[i])
        end
        write(file, "convparams", convparams)
    end
    open(string(dir,"info.txt"), "a+") do f
        seek(f, endpos)
        write(f,"run completed at $(now())<<<<<<<<<<<<<<<<<<<<\n")
        write(f,string("run time : ",now()-tstart,"\n"))
        println(string("run time : ",now()-tstart))
        for i in 1:numconvparams
            for j in 1:numconvobs
                ilowconval = findfirst(x -> x < thresh, deltas[j][i])
                if ilowconval != nothing
                    lowconval = convparams[i][2][ilowconval]
                    write(f, string(convobs[j][1]," converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh,"\n"))
                    println(string(convobs[j][1]," converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh))
                else
                    write(f, string(convobs[j][1]," not converged in ",convparams[i][1]," with threshold of ",thresh,"\n"))
                    println(string(convobs[j][1]," not converged in ",convparams[i][1]," with threshold of ",thresh))
                end
            end
        end
    end

    default(size = (800,600), reuse = true, title=unid, legend = true)
    plts = Vector{Any}(undef, numconvobs)
    for i in 1:numconvobs
        subplts=[]
        subplts = [plot([dt*i for i=0:numsteps], convdat[i,l], label=reshape([string(convparams[l][1]," = ", convparams[l][2][j]) for j in length(convparams[l][2]):-1:1], 1, length(convparams[l][2]))) for l in 1:numconvparams]
        plt = plot(subplts..., layout=(numconvparams,1), xlabel="t", ylabel=convobs[i][1])
        if saveplot
            savefig(plt, string(dir,"convplot_",convobs[i][1],"_",unid,".pdf"))
        end
        plts[i] = plt
    end

    return Bret, Dict([(convobs[i][1], convdat[i,:]) for i in 1:numconvobs]), dat, Dict([(convobs[i][1], plts[i]) for i in 1:numconvobs]), Dict([(convobs[i][1], deltas[i]) for i in 1:numconvobs]), unid
end

function convtdvp(dt, T, B::Function, H::Function; thresh=DEFCONVTHRESH, convparams=[], observables=[], convobs=nothing, verbose=false, timed=false, lightcone=false, multithread=false)

    if multithread
        return convtdvp_multithread(dt, T, B, H; thresh=thresh, convparams=convparams, observables=observables, convobs=convobs, lightcone=lightcone)
    end

    if convobs==nothing
        throw(error("convobs must be specified"))
    end

    numobs = length(observables)
    numconvparams = length(convparams)
    numconvobs = length(convobs)

    obkeys = [observables[i][1] for i=1:numobs]
    convobkeys = [convobs[i][1] for i=1:numconvobs]

    for i in 1:numconvobs
        if !(in(convobkeys[i], obkeys))
            push!(observables, convobs[i])
        end
    end
    numobs = length(observables)
    obkeys = [observables[i][1] for i=1:numobs]

    F = nothing
    t = 0
    numsteps = Int64(round(T/dt))

    convdat=fill(zeros(numsteps+1, 0), numconvobs, numconvparams)
    tstart=now()
    for i in 1:numconvparams
        numvals = length(convparams[i][2])
        vals = [convparams[i][2][end] for i=1:numconvparams]
        for j in numvals-1:-1:1
            vals[i] = convparams[i][2][j]
            for k in 1:numconvparams
                print(convparams[k][1]," : ",vals[k]," ")
            end
            println()
            dat=runtdvp!(dt, T, B(vals...), H(vals...), observables=[[convobs[i][1], psi->convobs[i][2](psi, vals...)] for i in 1:numconvobs], verbose=verbose, timed=timed, lightcone=lightcone)[2]
            for k=1:numconvobs
                convdat[k,i] = hcat(convdat[k,i], dat[convobs[k][1]])
            end
        end
    end
    vals = [convparams[i][2][end] for i=1:numconvparams]
    for k in 1:numconvparams
        print(convparams[k][1]," : ",vals[k]," ")
    end
    println()

    Bret, dat = runtdvp!(dt, T, B(vals...), H(vals...), observables=observables, verbose=verbose, timed=timed, lightcone=lightcone)

    for i in 1:numconvparams
        for j in 1:numconvobs
            convdat[j,i] = hcat(dat[convobs[j][1]], convdat[j,i])
        end
    end

    deltas = [[[rmsd(convdat[k,i][:,1], convdat[k,i][:,j]) for j in length(convparams[i][2]):-1:2] for i in 1:numconvparams] for k in 1:numconvobs]

    println(string("run time : ",now()-tstart))

    for i in 1:numconvparams
        for j in 1:numconvobs
            ilowconval = findfirst(x -> x < thresh, deltas[j][i])
            if ilowconval != nothing
                lowconval = convparams[i][2][ilowconval]
                println(string(convobs[j][1]," converged at ",convparams[i][1]," = ",lowconval," with threshold of ",thresh))
            else
                println(string(convobs[j][1]," not converged in ",convparams[i][1]," with threshold of ",thresh))
            end
        end
    end

    default(size = (800,600), reuse = true, title="", legend = true)
    plts = Vector{Any}(undef, numconvobs)
    for i in 1:numconvobs
        subplts=[]
        subplts = [plot([dt*i for i=0:numsteps], convdat[i,l], label=reshape([string(convparams[l][1]," = ", convparams[l][2][j]) for j in length(convparams[l][2]):-1:1], 1, length(convparams[l][2]))) for l in 1:numconvparams]
        plt = plot(subplts..., layout=(numconvparams,1), xlabel="t", ylabel=convobs[i][1])
        plts[i] = plt
    end

    return Bret, Dict([(convobs[i][1], convdat[i,:]) for i in 1:numconvobs]), dat, Dict([(convobs[i][1], plts[i]) for i in 1:numconvobs]), Dict([(convobs[i][1], deltas[i]) for i in 1:numconvobs])#, unid

end
