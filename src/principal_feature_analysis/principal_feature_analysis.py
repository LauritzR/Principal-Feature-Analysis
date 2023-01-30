# Copyright with the authors of the publication "A principal feature analysis"

import networkx as nx
import numpy as np
import scipy.stats
import random

# see paper Algorithm 2
def principal_feature_analysis(cluster_size,data,number_output_functions,freq_data,l,left_features,alpha,shuffle_feature_numbers):
    number_nodes= len(left_features) - number_output_functions  # Subtract the number of components of the output function
    list_of_nodes= left_features[number_output_functions:].copy()  # Take only the features and not the components of the output function
    m= data.shape[0]    # number of rows of the data matrix
    n = data.shape[1]   # number of columns of the data matrix
    number_chisquare_tests=0    # number of total chi-square tests
    counter_bin_less_than5 = 0  # number of chi-square tests with a bin less than 5 data points
    counter_bin_less_than1 = 0  # number of chi-square tests with a bin less than 1 data point
    global_adjm=np.zeros((m,m)) # global adjacency matrix
    is_entry_calculated=np.zeros((m,m)) # flag if the corresponding entry of the global adjacency matrix is already calculated
    while(True):
        print("Nodes left: " + str(number_nodes))
        list_of_clusters=[]
        intermediate_list=[]
        if shuffle_feature_numbers==1:
            random.shuffle(list_of_nodes)  # Pick the nodes of the subgraphs randomly if uncommented
        # Cluster the nodes into subsets of nodes with at most cluster_size nodes
        for i in range(0,int(number_nodes/cluster_size)+1):
            for j in range(0,cluster_size):
                if (i * cluster_size + j) < number_nodes:
                    intermediate_list.append(list_of_nodes[i * cluster_size + j])
            if intermediate_list != []:
                list_of_clusters.append(sorted(intermediate_list))
                intermediate_list=[]
        list_of_nodes=[]

        any_cluster_dissected=0  #  flag to notify if no further node was removed from any subgraph

        # for all clusters...
        for cluster in list_of_clusters:
            number_elements_cluster = len(cluster)
            counter_calculations = 0

            # ...calculate adjacency matrix using chi-square test...
            for i in range(0, number_elements_cluster):
                for j in range(i + 1, number_elements_cluster):
                    # if a chi-square test for the corresponding pair of nodes is already performed, the result from the last calculation is used, else the chi-square test is performed
                    # if both tested random features have more than one output value in their discretization because in this case the features are always independent
                    if is_entry_calculated[cluster[i],cluster[j]]==0 and (len(freq_data[cluster[i]])>1 and len(freq_data[cluster[j]])>1):
                        is_entry_calculated[cluster[i],cluster[j]]=1
                        is_entry_calculated[cluster[j],cluster[i]]=1
                        number_chisquare_tests +=1
                        freq_data_product = np.histogram2d(data[cluster[i], :], data[cluster[j], :],
                                                       bins=(l[cluster[i]], l[cluster[j]]))[0]
                        expfreq = np.outer(freq_data[cluster[i]], freq_data[cluster[j]]) / n

                        if sum(expfreq.flatten() < 5) > 0:
                            counter_bin_less_than5 += 1
                        if sum(expfreq.flatten() < 1) > 0:
                            counter_bin_less_than1 += 1
                        pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=-1)[1]
                        # ddof=-1 to have the degrees of freedom of the chi square eaual the number of bins, see corresponding paper (Appendix) for details
                        # if p-value pv is less than alpha the hypothesis that j is independent of i is rejected
                        if pv <= alpha:
                            global_adjm[cluster[i], cluster[j] ] = 1
                            global_adjm[cluster[j], cluster[i]] = 1
                    counter_calculations += 1
            adjm=(global_adjm[cluster,:])[:,cluster] # add the entries of the adjacency matrix of the subcluster to the global adjacency matrix
            
            # see paper Algorithm 1
            # ...create subgraphs based on adjacency matrix
            G=nx.from_numpy_array(adjm)
            S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

            list_graphs_to_divide=[]    # list of graphs to divide
            list_complete_sub_graphs=[] # list of complete subgraphs
            list_nodes_complete_sub_graphs=[]   # list of lists of nodes corresponding to the complete subgraphs of list_complete_sub_graphs

            # filter non-complete subgraphs
            for i in S:
                if list(nx.complement(i).edges)!=[]: # if a graph is not complete
                    list_graphs_to_divide.append(i)
                else:
                    list_complete_sub_graphs.append(i)
                    list_nodes_complete_sub_graphs.append(list(i.nodes))
            # remove nodes from non-complete subgraphs until only complete subgraphs are left
            while list_graphs_to_divide!=[]:
                any_cluster_dissected=1
                for current_graph in list_graphs_to_divide:
                    set_nodes_to_delete=nx.minimum_node_cut(current_graph)  # obtain a set of minimal cardinality to dissect the corresponding graph with a minimum cut algorithm
                    print(str(len(set_nodes_to_delete)) + " nodes removed!")
                    list_graphs_to_divide.remove(current_graph)             # remove the dissected graph from the list of graphs to divide
                    for node in list(set_nodes_to_delete):
                        current_graph.remove_node(node)                     # remove the nodes that were found with the minimum cut algorithm
                    list_new_sub_graphs = [current_graph.subgraph(c).copy() for c in nx.connected_components(current_graph)]
                    # Sort the new subgraphs into a list of complete subgraphs and subgraphs that can be further divided
                    for sub_graph_of_current_graph in list_new_sub_graphs:
                        if list(nx.complement(sub_graph_of_current_graph).edges)!=[]:
                            list_graphs_to_divide.append(sub_graph_of_current_graph)
                        else:
                            list_complete_sub_graphs.append(sub_graph_of_current_graph)
                            list_nodes_complete_sub_graphs.append(list(sub_graph_of_current_graph.nodes))

            # Transform the numbering of the nodes of the subcluster back to the global numbering
            for sub_list in list_nodes_complete_sub_graphs:
                for node_of_sub_list in sub_list:
                    list_of_nodes.append(cluster[node_of_sub_list])
        list_of_nodes.sort()
        number_nodes=len(list_of_nodes)     # Update the number of left features
        if any_cluster_dissected==0:    # If no subgraph was dissected set cluster_size = number_nodes to consider the total graph of the remaining features at once
            cluster_size=number_nodes
        if len(list_of_clusters)<=1:    # If the total graph of all features has been considered stop
            break
    print("Nodes left: " + str(number_nodes))
    print('Dissection done!')
    return list_nodes_complete_sub_graphs, counter_bin_less_than5 / number_chisquare_tests * 100, counter_bin_less_than1 / number_chisquare_tests * 100
