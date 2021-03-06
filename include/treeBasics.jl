mutable struct TreeNode
    parent::Int
    children::Vector{Int}
end

mutable struct Tree
    nodes::Vector{TreeNode}
end
Tree() = Tree([TreeNode(0, Vector{Int}())])
function Tree(len::Int)
    tree = Tree()
    for i=1:len-1
        addchild!(tree, i)
    end
    return tree
end

mutable struct TreeNetwork{T}
    tree::Tree
    sites::Vector{T}
end
TreeNetwork(sites::Vector{T}) where {T} = TreeNetwork{T}(Tree(length(sites)), sites)

function addchild!(tree::Tree, id::Int)
    1 <= id <= length(tree) || throw(BoundsError(tree, id))
    push!(tree.nodes, TreeNode(id, Vector{}()))
    child = length(tree)
    push!(tree.nodes[id].children, child)
    return tree
end
function addchild!(network::TreeNetwork, id::Int, site)
    addchild!(network.tree, id)
    push!(network.sites, site)
    return network
end
function removechild!(tree::Tree, id::Int)
    loopcheck(tree)
    1 <= id <= length(tree) || throw(BoundsError(tree, id))
    parid = tree[id].parent
    parid == 0 && throw(ErrorException("attempt to remove the head-node"))
    for child in tree[id].children
        removechild!(tree, child)
    end
    filter!(x->x!=id, tree[parid].children)
    deleteat!(tree.nodes, id)
    for (i, node) in enumerate(tree)
        if node.parent > id
            tree[i].parent -= 1
        end
        for (j, child) in enumerate(node.children)
            if (child > id)
                tree[i].children[j] -= 1
            end
        end
    end
    return tree
end

#adds tree2 to tree1 at node id of tree1
function addtree(tree1::Tree, id::Int, tree2::Tree)
    t1=deepcopy(tree1)
    t2=deepcopy(tree2)
    len1 = length(t1)
    push!(t1[id].children, len1+1)
    t2[1].parent = id
    t2[1].children .+= len1
    for i=2:length(t2)
        t2[i].parent += len1
        t2[i].children .+= len1
    end
    Tree([t1.nodes..., t2.nodes...])
end
#same as above but modifies tree1
function addtree!(tree1::Tree, id::Int, tree2::Tree)
    t2=deepcopy(tree2)
    len1 = length(tree1)
    push!(tree1[id].children, len1+1)
    t2[1].parent = id
    t2[1].children .+= len1
    for i=2:length(t2)
        t2[i].parent += len1
        t2[i].children .+= len1
    end
    push!(tree1.nodes, t2.nodes...)
    return tree1
end

function addtree(net1::TreeNetwork, id::Int, net2::TreeNetwork)
    TreeNetwork(addtree(net1.tree, id, net2.tree), [net1.sites..., net2.sites...])
end
function addtree!(net1::TreeNetwork, id::Int, net2::TreeNetwork)
    addtree!(net1.tree, id, net2.tree)
    push!(net1.sites, net2.sites...)
    return net1
end

function bonds(tree::Tree)
    N = length(tree)
    h = findheadnode(tree)
    r = []
    for i=1:h-1
        push!(r, (tree[i].parent, i))
    end
    for i=h+1:N
        push!(r, (tree[i].parent, i))
    end
    r
end
bonds(net::TreeNetwork) = bonds(net.tree)

function bondview(tree::Tree)
    N = length(tree)
    mat = zeros(Int, N, N)
    for bond in bonds(tree)
        mat[bond...] = 1
        mat[reverse(bond)...] = 1
    end
    mat
end
bondview(net::TreeNetwork) = bondview(net.tree)

#finds the leaves of a tree, ie the sites which are only connected to one other site
function leaves(tree::Tree)
    r=[]
    for (i, node) in enumerate(tree)
        if length(node.children)==0 || (length(node.children)==1 && node.parent==0)
            push!(r, i)
        end
    end
    r
end
leaves(net::TreeNetwork) = leaves(net.tree)

#return list of node ids starting with id and ending with the head-node such that each element is the parent of the one to its left
function pathtohead(tree::Tree, id::Int)
    path = [id]
    hn = findheadnode(tree)
    i=id
    while i != hn
        i = tree[i].parent
        push!(path, i)
    end
    return path
end
pathtohead(net::TreeNetwork, id::Int) = net[pathtohead(net.tree, id)]

function pathfromhead(tree::Tree, id::Int)
    path = [id]
    hn = findheadnode(tree)
    i=id
    while i != hn
        i = tree[i].parent
        push!(path, i)
    end
    return reverse(path)
end
pathfromhead(net::TreeNetwork, id::Int) = net[pathfromhead(net.tree, id)]

function loopcheck(tree::Tree)
    len = length(tree)
    ids = [findheadnode(tree)]
    for i=1:len
        for child in tree[i].children
            in(child, ids) && throw(ErrorException("loop found in tree!"))
            push!(ids, child)
        end
    end
end
loopcheck(net::TreeNetwork) = loopcheck(net.tree)

function findheadnode(tree::Tree)
    for i in 1:length(tree)
        if tree[i].parent==0
            return i
        end
    end
end
findheadnode(net::TreeNetwork) = findheadnode(net.tree)

function findchild(children::Vector{Int}, id::Int)
    return findfirst(x->x==id, children)
end

function setheadnode!(tree::Tree, id::Int)
    par = tree[id].parent
    if par != 0
        #set the parent to the head-node
        setheadnode!(tree, par)
        #println("setting parent of $par to $id")
        tree[par].parent=id
        #println("removing $id as child of $(par)")
        filter!(x->x!=id, tree[par].children)
        #println("adding $par as child of $id")
        pushfirst!(tree[id].children, par)
        #println("setting parent of $id to 0")
        tree[id].parent=0
    end
end
function setheadnode!(net::TreeNetwork, id::Int)
    par = net.tree[id].parent
    if par != 0
        #set the parent to the head-node
        setheadnode!(net, par)
        #println("setting parent of $par to $id")
        net.tree[par].parent=id
        #println("removing $id as child of $par")
        childpos=findchild(net.tree[par].children, id)
        filter!(x->x!=id, net.tree[par].children)
        #println("adding $par as child of $id")
        pushfirst!(net.tree[id].children, par)
        #println("setting parent of $id to 0")
        net.tree[id].parent=0

        Apar = net[par]
        dpar = size(Apar)
        #println("remove dummy index of old head-node")
        Apar = reshape(Apar, dpar[2:end])
        nd = ndims(Apar)
        IA = collect(1:nd)
        IC = collect(1:nd)
        deleteat!(IC, childpos)
        pushfirst!(IC, childpos)
        net[par] = tensorcopy(Apar, IA, IC)
        #println("add dummy index to new head-node")
        Ahead = net[id]
        net[id] = reshape(Ahead, 1, size(Ahead)...)        
    end
end
function setheadnode(tree_::Tree, id::Int)
    tree = deepcopy(tree_)
    setheadnode!(tree, id)
    return tree
end
function setheadnode(net_::TreeNetwork, id::Int)
    net = deepcopy(net_)
    setheadnode!(net, id)
    return net
end

"""
Tree iterators:
"""
struct Radial{T}
    iter::T
    start::Int
end
Radial(iter) = Radial(iter, findheadnode(iter))
struct Walk{T}
    iter::T
end
struct Traverse{T}
    iter::T
end

Base.length(tree::Tree) = length(tree.nodes)
Base.length(net::TreeNetwork) = length(net.tree.nodes)
Base.iterate(iter::Tree, state) = iterate(iter.nodes, state)
Base.iterate(iter::Tree) = iterate(iter.nodes)
Base.iterate(iter::TreeNetwork, state) = iterate(iter.sites, state)
Base.iterate(iter::TreeNetwork) = iterate(iter.sites)
Base.getindex(iter::Tree, i::Union{Int,Vector{Int},UnitRange}) = getindex(iter.nodes, i)
Base.getindex(iter::TreeNetwork, i::Union{Int,Vector{Int},UnitRange}) = getindex(iter.sites, i)

function Base.getindex(iter::Union{Tree,TreeNetwork}, I::Tuple{Int,Int})
    1 <= I[2] <= length(iter) || throw(BoundsError(iter, I[2]))
    1 <= I[1] <= length(iter) || throw(BoundsError(iter, I[1]))
    tmp = setheadnode(iter, I[2])
    return pathtohead(tmp, I[1])
end

Base.setindex!(iter::Tree, v::TreeNode, i) = setindex!(iter.nodes, v, i)
Base.setindex!(iter::TreeNetwork{T}, v::T, i) where {T} = setindex!(iter.sites, v, i)
Base.firstindex(iter::Union{Tree,TreeNetwork}) = 1
Base.lastindex(iter::Union{Tree,TreeNetwork}) = length(iter)
Base.eltype(iter::Tree) = TreeNode
Base.eltype(iter::TreeNetwork{T}) where {T} = T
Base.eltype(iter::Union{Walk{Tree},Traverse{Tree}}) = Int
Base.eltype(iter::Radial{Tree}) = Vector{Int}
Base.eltype(iter::Union{Walk{TreeNetwork{T}},Traverse{TreeNetwork{T}}}) where {T} = Tuple{Int,T}
Base.eltype(iter::Radial{TreeNetwork{T}}) where {T} = Vector{Tuple{Int,T}}

Base.length(iter::Traverse) = length(iter.iter)
function Base.length(iter::Union{Walk,Radial})
    len=0
    for i in iter
        len += 1
    end
    return len
end
function Base.getindex(iter::Union{Radial,Walk,Traverse}, I::Int)
    1 <= I <= length(iter) || throw(BoundsError(iter, I))
    for (i, val) in enumerate(iter)
        i==I && return val
    end
end
function Base.getindex(iter::Union{Radial,Walk,Traverse}, I::UnitRange{Int})
    1 <= I.start <= length(iter) || throw(BoundsError(iter, I))
    1 <= I.stop <= length(iter) || throw(BoundsError(iter, I))
    len = I.stop-I.start+1
    r = Vector{eltype(iter)}(undef, len)
    i=1
    (val, state) = iterate(iter)
    while i < I.start
        (val, state) = iterate(iter, state)
        i+=1
    end
    r[i-I.start+1] = val
    while i < I.stop
        i+=1
        (val, state) = iterate(iter, state)
        r[i-I.start+1] = val
    end
    return r
end
Base.lastindex(iter::Union{Radial,Walk,Traverse}) = length(iter)
Base.firstindex(iter::Union{Radial,Walk,Traverse}) = 1

function Base.iterate(iter::Radial{Tree}, state::Vector{Int})
    children = state
    length(children) == 0 && return nothing
    tree = iter.iter
    grandchildren = cat(map(x->x.children, tree[children])..., dims=1)
    return (deepcopy(children), grandchildren)
end
function Base.iterate(iter::Radial{Tree})
    tree = iter.iter
    hn = iter.start
    children = tree[hn].children
    return ([hn], children)
end

function Base.iterate(iter::Radial{TreeNetwork{T}}, state::Vector{Int}) where {T}
    net = iter.iter
    start = iter.start
    next = iterate(Radial(net.tree, start), state)
    if next==nothing
        return nothing
    else
        (i, state) = next
        return ([(i[k],net[i[k]]) for k=1:length(i)], state)
    end
end
function Base.iterate(iter::Radial{TreeNetwork{T}}) where {T}
    net = iter.iter
    start = iter.start
    (i, state) = iterate(Radial(net.tree, start))
    return ([(i[k],net[i[k]]) for k=1:length(i)], state)
end

function Base.iterate(iter::Walk{Tree}, state::Tuple{Tree,Int})
    tree = state[1]
    id = state[2]
    children = tree[id].children
    par = id
    if length(children) == 0
        par = tree[par].parent
        par == 0 && return nothing
        popfirst!(tree[par].children)
        return (par, (tree, par))
    end
    child = tree[par].children[1]
    return (child, (tree, child))
end
function Base.iterate(iter::Walk{Tree})
    tree_ = iter.iter
    hn = findheadnode(tree_)
    tree = deepcopy(tree_)
    return (hn, (tree, hn))
end

function Base.iterate(iter::Walk{TreeNetwork{T}}, state::Tuple{Tree,Int}) where {T}
    net = iter.iter
    next = iterate(Walk(net.tree), state)
    if next==nothing
        return nothing
    else
        (i, state) = next
        return ((i, net[i]), state)
    end
end
function Base.iterate(iter::Walk{TreeNetwork{T}}) where {T}
    net = iter.iter
    (i, state) = iterate(Walk(net.tree))
    return ((i, net[i]), state)
end

function Base.iterate(iter::Traverse{Tree}, state::Tuple{Tree,Int})
    tree = state[1]
    id = state[2]
    children = tree[id].children
    par = id
    while length(children) == 0
        par = tree[par].parent
        par == 0 && return nothing
        popfirst!(tree[par].children)
        children = tree[par].children
    end
    child = tree[par].children[1]
    return (child, (tree, child))
end
function Base.iterate(iter::Traverse{Tree})
    tree_ = iter.iter
    hn = findheadnode(tree_)
    tree = deepcopy(tree_)
    return (hn, (tree, hn))
end

function Base.iterate(iter::Traverse{TreeNetwork{T}}, state::Tuple{Tree,Int}) where {T}
    net = iter.iter
    next = iterate(Traverse(net.tree), state)
    if next==nothing
        return nothing
    else
        (i, state) = next
        return ((i, net[i]), state)
    end
end
function Base.iterate(iter::Traverse{TreeNetwork{T}}) where {T}
    net = iter.iter
    (i, state) = iterate(Traverse(net.tree))
    return ((i, net[i]), state)
end

import Base: print, println, show
function print(tree::Tree, id::Int)
    loopcheck(tree)
    print(id)
    nochild = length(tree[id].children)
    if nochild == 1
        print("->")
        print(tree, tree[id].children[1])
    elseif nochild > 1
        print("->")
        printstyled("(",color=:yellow)
        for child in tree[id].children[1:end-1]
            print(tree, child)
            printstyled(";", color=:green)
        end
        print(tree, tree[id].children[end])
        printstyled(")", color=:yellow)
    end
end
print(tree::Tree) = print(tree, findheadnode(tree))
println(tree::Tree) = (print(tree); println())
print(net::TreeNetwork) = (println(net.tree); print(net.sites))
println(net::TreeNetwork) = (print(net); println())

show(io::IO, tree::Tree) = println(tree)
show(io::IO, net::TreeNetwork) = println(net)
