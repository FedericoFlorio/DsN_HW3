# Utilities

function adjacency(W::Array{Float64})
    return [(W[i,j]>0 ? 1 : 0) for i in 1:size(W)[1], j in 1:size(W)[2]]
end

function neig_out(x::Int,A::Array{Int})
    return [i for i in 1:size(A)[2] if A[x,i] == 1]
end
function neig_out(x::Int,A::Array{Float64})
    return [i for i in 1:size(A)[2] if A[x,i] > 0]
end

function neig_in(x::Int,A::Array{Int})
    return [i for i in 1:size(A)[1] if A[x,i] == 1]
end
function neig_in(x::Int,A::Array{Float64})
    return [i for i in 1:size(A)[1] if A[x,i] > 0]
end



# Types and constructors

struct Graph
    N::Int                          # dimension (number of nodes)
    nodes::Vector{Int}              # list of nodes
    edges::Vector{Tuple{Int,Int}}   # list of edges
    A::Matrix{Int}                  # adjacency matrix
    W::Matrix{Float64}              # weight matrix
    w_out::Vector{Float64}          # out degrees of nodes
    w_in::Vector{Float64}           # in degrees of nodes
    P::Matrix{Float64}              # normalized weight matrix
end

function Graph(W)
    size(W)[1] != size(W)[2] && error("The weight matrix must be a square matrix")
    N = size(W)[1]
    V = Vector(1:N)
    A = adjacency(W)
    w_out = [sum(W[i,:]) for i in 1:N]
    w_in = [sum(W[:,i]) for i in 1:N]
    P = [W[i,j]/w_out[i] for i in 1:N, j in 1:N]
    E = [(i,j) for i in 1:N, j in 1:N if A[i,j] == 1]

    return Graph(N,V,E,A,W,w_out,w_in,P)
end



# Centralities

function centrality_degree(G::Graph)
    return G.w_in/sum(G.w_in)
end

function centrality_eig(G::Graph)
    Λ = eigen(G.W')
    z = Λ.vectors[:,argmax(abs.(Λ.values))]
    return z./sum(z)
end

function centrality_inv_dist(G::Graph)
    Λ = eigen(G.P')
    z = Float64.(Λ.vectors[:,argmax(real.(Λ.values))])
    return z./sum(z)
end

function centrality_katz(G::Graph,β::Float64,μ::Vector{Float64})
    λ = eigmax(W)
    z = (I - (1-β)/λ*W') \ (β*μ)
    return z/sum(z)
end

function pagerank(G::Graph, β::Float64, μ::Vector{Float64}; maxt=100, eps=1e-2)
    pr = copy(μ)
    for t in 1:maxt
        pr_old = copy(pr)
        for i in G.nodes
            pr[i] = β*μ[i] + (1-β)*dot(G.P[:,i],pr_old)
            pr /= sum(pr)
        end
        norm(pr-pr_old)/norm(pr_old) ≤ eps && break
    end
    return pr
end