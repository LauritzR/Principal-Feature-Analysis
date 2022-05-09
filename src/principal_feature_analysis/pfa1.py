# Algorithm 1 of PFA with :
# 1. Adjacency Matrix based on arbitrary definition of correlation and
# 2. no association of X with Y/output is used (not supervised).

# There are reasons why these points are not considered in the
# original implementations. These reasons are mentioned in the paper.

# Particularly, tightly connected dependency graphs are not pruned (as
# in test_data0). This does not mean that we cannot prune the nodes
# based on redundancy or functional dependency, but rather that such
# pruning is not unique and arbitrary removal would suffice.

import networkx as nx
import numpy as np
import scipy.stats
from itertools import combinations
from random import sample, seed

def test_data0(n=5000):
    # Everything is a function of everything. Nothing gets deleted by PFA.
    A = 5*np.random.rand(n, 3)

    A[:, 1] = 0.6*A[:, 0] # proportional
    A[:, 2] = 0.3*A[:, 0] + 1.5*A[:, 1] # linear combination
    return A

def test_data1(n=5000):
    # Ex 3 from the paper
    A = 5*np.random.rand(n, 5)

    A[:, 3] = 2*A[:, 0]*A[:, 1]*A[:, 2]
    A[:, 4] = A[:, 0]*A[:, 1]
    return A    

def test_data2(n=5000):
    # Test data from the package: 0 & 1 are independent
    A=5*np.random.rand(n, 5)

    A[:,2]=A[:,0]*0.01 + 5
    A[:,3]=A[:,0]*A[:,1]**2 # this serves as a bridge between ind islands 0 & 1
    A[:,4]=np.exp(-A[:,1])

    return A

def test_data3(n=5000):
    # Test data from the package: 0 & 1 are independent
    A=5*np.random.rand(n, 5)

    A[:,2]=A[:,0]*0.01 + 5
    A[:,3]=A[:,1]**2
    A[:,4]=np.exp(-A[:,1])

    return A    

def ex_cor_fun(x, y, alt='two-sided'):
    # Example of a custom correlation function which will work with cor_mat
    return scipy.stats.kendalltau(x, y, variant='c', alternative=alt)

def cor_mat(X, meth="p", **kwargs):
    """
    X -- data
    meth -- 'p' for Pearson, 's' for Spearman, 'k' for Kendall or
            a callable that calculates correlation and p-val from two signals
    """
    n = X.shape[1]
    C = np.zeros((n, n)) # container for cor coef, may be optimized to be sparse
    P = np.ones((n, n)) # container for cor P-val, may be optimized to be sparse
    cmb = combinations(range(n), 2)
    if(hasattr(meth, '__call__')):
        cor_fun = meth
    elif(isinstance(meth, str)):
        if(meth == 'p'):
            cor_fun = scipy.stats.pearsonr
        elif(meth == 's'):
            cor_fun = scipy.stats.spearmanr
        elif(meth == 'k'):
            cor_fun = scipy.stats.kendalltau
        else:
            raise ValueError ("Unknown symbol %s" % meth)
    else:
        raise ValueError("Unknown type of method")
            
    for c in cmb:
        cor, pval = cor_fun(X[:, c[0]], X[:, c[1]], **kwargs)
        C[c[0], c[1]] = cor
        P[c[0], c[1]] = pval

    return C, P

def cor_adj_mat(X, meth='p', alpha=0.05, correct=False, **kwargs):
    C, P = cor_mat(X, meth=meth, **kwargs)
    if correct: # simple P-val correction
        n = C.shape[1] # C/P must be square upper triangular
        n_cmb = n*(n-1) / 2
        print("N. comparisons:", str(n_cmb))
        P = P * n_cmb
    return P < alpha

def cor_graph(cor_adj_mat):
    return nx.from_numpy_matrix(cor_adj_mat)

def pfa1_full(X, meth='p', alpha=0.05, correct=True, rnd_seed=None, **kwargs):
     adj = cor_adj_mat(X, meth=meth, alpha=alpha, correct=correct, **kwargs)
     print("Adjacency matrix:")
     print(adj)
     gr = cor_graph(adj)
     subgr, subgr_nodes, subgr_edges = pfa1(gr, rnd_state=rnd_seed)
     return subgr, subgr_nodes, subgr_edges

def pfa1(graph, rnd_state=None):
    """
    Core Algorithm 1 of PFA.
    """
    seed(rnd_state) # seed rnd number generator, if None, then not reproducible
    S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    nS = len(S)
    print("N. cc:", str(nS))
    
    list_graphs_to_divide=[] # list of graphs to divide
    list_complete_sub_graphs=[] # list of complete subgraphs
    list_nodes_complete_sub_graphs=[] # list of lists of nodes corresponding to the complete subgraphs of list_complete_sub_graphs

    # filter non-complete subgraphs
    for i in sample(S, nS):
        if list(nx.complement(i).edges)!=[]: # if a graph is not complete
            list_graphs_to_divide.append(i)
        else:
            list_complete_sub_graphs.append(i)
            list_nodes_complete_sub_graphs.append(list(i.nodes))
    n_iter = 1
    # remove nodes from non-complete subgraphs until only complete subgraphs are left

    while list_graphs_to_divide!=[]:
        print("Iteration: " + str(n_iter), end="\r")
        # any_cluster_dissected=1
        for current_graph in list_graphs_to_divide:
            set_nodes_to_delete=nx.minimum_node_cut(current_graph)  # minimum cut algorithm
            # print(str(len(set_nodes_to_delete)) + " node(s) removed:")
            # print(set_nodes_to_delete)
            # print(" from "+str(current_graph.nodes)+" graph nodes")
            list_graphs_to_divide.remove(current_graph) # remove the dissected graph
            for node in list(set_nodes_to_delete):
                current_graph.remove_node(node) # remove the minimum cut nodes
            new_S = [current_graph.subgraph(c).copy() for c in nx.connected_components(current_graph)]
            # Sort the new subgraphs into a list of complete subgraphs and subgraphs that can be further divided
            for sub_graph_of_current_graph in new_S:
                if list(nx.complement(sub_graph_of_current_graph).edges)!=[]:
                    list_graphs_to_divide.append(sub_graph_of_current_graph)
                else:
                    list_complete_sub_graphs.append(sub_graph_of_current_graph)
                    list_nodes_complete_sub_graphs.append(list(sub_graph_of_current_graph.nodes))
        n_iter = n_iter + 1

    print("N. iterations:",str(n_iter-1))
    n = len(list_complete_sub_graphs)
    print("N. subgraphs:", str(n))
    sub_graph_components = [list(x.nodes) for x in list_complete_sub_graphs]
    sub_graph_arch = [list(x.edges) for x in list_complete_sub_graphs]
    return list_complete_sub_graphs, sub_graph_components, sub_graph_arch



if __name__ == "__main__":
    print("===================================================")
    print("Ex. 1: tightly connected graph, nothing gets pruned")
    A = test_data0()
    x, y, z = pfa1_full(A)

    print("Sub graphs:")
    print(y)

    print("===================================================")
    print("Ex. 2: Example 3 from the paper")
    A = test_data1()
    x, y, z = pfa1_full(A)

    print("Sub graphs:")
    print(y)

    print("===================================================")
    print("Ex. 3: Test data from the package, v0 & v11 are fully independent, hierarchy via v3")
    A = test_data2()
    x, y, z = pfa1_full(A)

    print("Sub graphs:")
    print(y)

    print("===================================================")
    print("Ex. 4: As Ex.3, v0 & v1 are fully independent, but v3 is not a bridge, no hierarchy")
    A = test_data3()
    x, y, z = pfa1_full(A)

    print("Sub graphs:")
    print(y)
