import ot
import numpy as np

def sparse_ot(weights1, weights2, M):
    """ Compute transport for (posssibly) un-normalized sparse distributions"""
    
    weights1 = weights1/weights1.sum()
    weights2 = weights2/weights2.sum()
    
    # Identifies the indices of non-zero elements in weights1 and weights2
    # to improve computation efficiency
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    # this should be 20 by 20 as it only use top 20 words to represent a topic
    # M is the cost matrix between embeddings M  = euclidean_distances(embeddings, embeddings) ** p
    # M_reduced represents the cost of transporting mass between the active elements of the two distributions.
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = np.ascontiguousarray(M[active1][:,active2])
    
    return ot.emd2(weights_1_active,weights_2_active,M_reduced)

def hott(p, q, C, threshold=None):
    """ Hierarchical optimal topic transport."""
    
    # Avoid changing p and q outside of this function
    p, q = np.copy(p), np.copy(q)
    
    k = len(p)
    if threshold is None:
        threshold = 1. / (k + 1)
        
    p[p<threshold] = 0
    q[q<threshold] = 0
    
    return sparse_ot(p, q, C)


def hoftt(p, q, C):
    """ Hierarchical optimal full topic transport."""
    return ot.emd2(p, q, C)

    





