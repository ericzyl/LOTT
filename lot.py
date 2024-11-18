import ot
import numpy as np

def makeGaussian1D(size, fwhm=3, center=None):
    """
    Generate a 1D Gaussian distribution.

    size: Length of the output vector.
    fwhm: Full-Width-Half-Maximum, defines the width of the Gaussian.
    center: Center of the Gaussian (if None, it will be centered in the middle of the vector).
    """
    # If center is not specified, default to the middle of the vector
    center = center if center is not None else size // 2

    # Generate the Gaussian distribution
    x = np.arange(size)
    gaussian = np.exp(-4 * np.log(2) * ((x - center)**2) / fwhm**2)
    
    # Normalize the Gaussian distribution
    return gaussian / gaussian.sum()


def makeBimodal1D(size, fwhm1=3, center1=None, fwhm2=3, center2=None):
    """
    Generate a 1D bimodal Gaussian distribution.

    size: Length of the output vector.
    fwhm1: Full-Width-Half-Maximum for the first peak.
    center1: Center of the first Gaussian peak (default: middle of the vector).
    fwhm2: Full-Width-Half-Maximum for the second peak.
    center2: Center of the second Gaussian peak (default: 3/4 of the vector).
    """
    # If centers are not specified, set them to default positions
    center1 = center1 if center1 is not None else size // 4  # Default at 1/4 position
    center2 = center2 if center2 is not None else 3 * size // 4  # Default at 3/4 position

    # Generate the 1D coordinate array
    x = np.arange(size)

    # Create two Gaussian distributions with specified centers and widths (fwhm)
    gaussian1 = np.exp(-4 * np.log(2) * ((x - center1)**2) / fwhm1**2)
    gaussian2 = np.exp(-4 * np.log(2) * ((x - center2)**2) / fwhm2**2)

    # Combine the two Gaussians to form a bimodal distribution
    bimodal = gaussian1 + gaussian2

    # Normalize the bimodal distribution
    return bimodal / bimodal.sum()



def lot(coupling_matrix, mydictionary):
    """
    input: optimal transport plan (k,k), where k is number of topics
    mydictionary is the lda centers (k,word_dim)
    output: LOT embedding
    """
    # create mapping matrix
    T_mu_sigma = []

    # Iterate over each point x in sigma to find the max transported mass
    # it should not be in shape[0], but need to look at the shape of coupling matrix
    # range(coupling_matrix.shape[0]) is 784
    # 300 is the dimension of word embedding
    for i in range(coupling_matrix.shape[0]):
        source_weight = coupling_matrix[i].sum()
        y_coor = np.zeros(300, dtype=float)
        if source_weight == 0:
            T_mu_sigma.append(y_coor)
        else:
            # # find barycentric projection
            # for j in range(coupling_matrix.shape[1]):
            #     # THIS PART CAN BE OPTIMIZED (checking the nonzero indices)
            #     weight = coupling_matrix[i][j]
            #     if weight > 0:
            #         y_coor += weight*np.array(mydictionary[j])
            # y_coor = y_coor/source_weight
            # T_mu_sigma.append(y_coor)

            # Get the indices of nonzero weights
            nonzero_indices = np.where(coupling_matrix[i] > 0)[0]
            
            # Use vectorized operations on nonzero indices only
            weights = coupling_matrix[i][nonzero_indices]
            positions = np.array([mydictionary[j] for j in nonzero_indices])
            
            # Calculate weighted sum for barycentric projection
            y_coor = (weights[:, None] * positions).sum(axis=0) / source_weight
            
            T_mu_sigma.append(y_coor)

        #print(np.array(T_mu_sigma).flatten().shape)

    return np.array(T_mu_sigma).flatten()


def create_lot_embeddings(samples, reference, mydictionary, distance, distance_metric='euclidean'):
    """
    samples is (m,k) where m is number of document and k is number of topics
    reference is chosen as gaussian distribution
    mydictionary is lda centers
    """
    lot_embeddings = []
    for i in range(len(samples)):
        # calculate distance matrix
        # assuming samples are all distributions
        s = samples[i].reshape(-1,1)
        # s = s/s.sum()
        r = reference.reshape(-1,1)
        M = distance #ot.dist(r, s, metric=distance_metric)*

        P_sigma_mu = ot.emd(s.reshape(-1,), reference.reshape(-1,), M)
        #print(P_sigma_mu.shape)
        lot_embeddings.append(lot(P_sigma_mu, mydictionary))

    return lot_embeddings

def lot_wmd_embeddings(samples, reference, word_embeddings, distance):
    lot_embeddings = []
    for i in range(len(samples)):
        s = samples[i]/samples[i].sum()
        print(len(s))
        print(word_embeddings.shape[0])
        active = np.where(s)[0]
        s_active = s[active]
        M_reduced = np.ascontiguousarray(distance[active])
        coupling = ot.emd(s_active.reshape(-1,), reference.reshape(-1,), M_reduced)


        # embeddings
        T_mu_sigma = [np.zeros(300) for _ in range(word_embeddings.shape[0])]
        # Compute values in a loop and assign them to the active indices
        for idx in active:
            for i in range(coupling.shape[0]):
                source_weight = coupling[i].sum()
                y_coor = np.zeros(300, dtype=float)
                if source_weight == 0:
                    T_mu_sigma[idx] = y_coor
                else:
                    # find barycentric projection
                    for j in range(coupling.shape[1]):
                        # THIS PART CAN BE OPTIMIZED (checking the nonzero indices)
                        weight = coupling[i][j]
                        if weight > 0:
                            y_coor += weight*np.array(word_embeddings[j])
                    y_coor = y_coor/source_weight
                    T_mu_sigma[idx] = y_coor

        lot_embeddings.append(np.array(T_mu_sigma).flatten())
    
    return lot_embeddings

