function inv_dist_continuous(G::Graph)
    L = diagm(G.w_out) - G.W
    π = nullspace(L')
    return π./sum(π)
end