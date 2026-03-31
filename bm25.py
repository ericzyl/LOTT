import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# BM25 core
# ---------------------------------------------------------------------------

class BM25:
    """BM25 over a bag-of-words matrix (documents × vocab)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    def fit(self, bow_matrix: np.ndarray):
        """bow_matrix: (n_docs, vocab_size) integer counts."""
        self.bow   = bow_matrix.astype(float)
        N, V       = self.bow.shape
        self.N     = N

        # document lengths and average
        self.dl    = self.bow.sum(axis=1)               # (N,)
        self.avgdl = self.dl.mean()

        # IDF:  log((N - df + 0.5) / (df + 0.5) + 1)
        df         = (self.bow > 0).sum(axis=0)         # (V,)
        self.idf   = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def score(self, query_bow: np.ndarray) -> np.ndarray:
        """
        query_bow: (vocab_size,) integer counts for one query document.
        Returns BM25 scores (N,) — higher is more similar.
        """
        q_terms = np.where(query_bow > 0)[0]
        scores  = np.zeros(self.N)

        for t in q_terms:
            tf_num = self.bow[:, t] * (self.k1 + 1)
            tf_den = self.bow[:, t] + self.k1 * (
                1 - self.b + self.b * self.dl / self.avgdl
            )
            scores += self.idf[t] * tf_num / tf_den

        return scores

    def score_matrix(self, query_bows: np.ndarray) -> np.ndarray:
        """
        query_bows: (n_queries, vocab_size)
        Returns (n_queries, N) score matrix.
        """
        return np.array([self.score(q) for q in query_bows])


# ---------------------------------------------------------------------------
# BM25 distance (for KNN — lower = more similar)
# ---------------------------------------------------------------------------

def bm25_distance(query_bow: np.ndarray,
                  doc_bow:   np.ndarray,
                  bm25_model: BM25) -> float:
    """
    Wrapper so BM25 fits the same (p, q, C) signature used in knn_classifier.
    Here C carries the fitted BM25 model instead of a cost matrix.
    BM25 gives similarity, so we negate it to get a distance.
    """
    score = bm25_model.score(query_bow)
    # doc_bow is one row — find its index by dot product identity trick
    # We pass the full score vector and index into it via doc identity
    # Instead: score just this one doc directly
    return -bm25_model.score(query_bow)[0]  # placeholder; see knn_bm25 below


# ---------------------------------------------------------------------------
# KNN with BM25  (replaces the generic knn() for BM25)
# ---------------------------------------------------------------------------

def knn_bm25(bow_train: np.ndarray,
             bow_test:  np.ndarray,
             y_train:   np.ndarray,
             y_test:    np.ndarray,
             k1: float = 1.5,
             b:  float = 0.75,
             n_neighbors: int = 7) -> float:
    """
    KNN classification using BM25 similarity.
    Fits BM25 on training docs, scores each test doc against all train docs.
    """
    from knn_classifier import predict

    bm25 = BM25(k1=k1, b=b).fit(bow_train)
    n_classes = len(np.unique(y_train))
    predictions = []

    for i, q in enumerate(bow_test):
        scores = bm25.score(q)            # (n_train,) — higher = more similar
        # Convert to distances for argsort (lower = better)
        distances = -scores
        rank      = np.argsort(distances)[:n_neighbors]
        predictions.append(predict(y_train[rank], n_classes))

    test_error = 1 - (np.array(predictions) == y_test).mean()
    return test_error


# ---------------------------------------------------------------------------
# BM25 + LOTT combination
# ---------------------------------------------------------------------------

def combine_bm25_lott(bm25_scores:   np.ndarray,
                      lott_distances: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """
    Combine BM25 similarity scores and LOTT distances into one distance.

    bm25_scores:    (n,) higher = more similar
    lott_distances: (n,) lower  = more similar  (euclidean on LOTT embeddings)
    alpha:          weight on BM25 side (1-alpha on LOTT side)

    Strategy: normalise both to [0,1], convert BM25 to a distance,
    then take a weighted sum.
    """
    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    bm25_dist  = 1.0 - _norm(bm25_scores)   # flip: high score → low distance
    lott_dist  = _norm(lott_distances)

    return alpha * bm25_dist + (1 - alpha) * lott_dist


# def knn_bm25_lott(bow_train:     np.ndarray,
#                   bow_test:      np.ndarray,
#                   lott_train:    np.ndarray,
#                   lott_test:     np.ndarray,
#                   y_train:       np.ndarray,
#                   y_test:        np.ndarray,
#                   alpha:         float = 0.5,
#                   k1:            float = 1.5,
#                   b:             float = 0.75,
#                   n_neighbors:   int   = 7) -> float:
#     """
#     KNN using a convex combination of BM25 and LOTT distances.

#     For each test doc:
#       - BM25 scores it against all training docs
#       - LOTT euclidean distance to all training docs
#       - Combined distance = alpha * BM25_dist + (1-alpha) * LOTT_dist
#     """
#     from knn_classifier import predict

#     bm25      = BM25(k1=k1, b=b).fit(bow_train)
#     n_classes = len(np.unique(y_train))
#     predictions = []

#     for i, (q_bow, q_lott) in enumerate(zip(bow_test, lott_test)):
#         # BM25 scores (n_train,)
#         bm25_scores = bm25.score(q_bow)

#         # LOTT euclidean distances (n_train,)
#         lott_dists  = np.linalg.norm(lott_train - q_lott, axis=1)

#         # Combined distance
#         combined = combine_bm25_lott(bm25_scores, lott_dists, alpha=alpha)

#         rank = np.argsort(combined)[:n_neighbors]
#         predictions.append(predict(y_train[rank], n_classes))

#     test_error = 1 - (np.array(predictions) == y_test).mean()
#     return test_error

# Vectorized Implementation
def knn_bm25_lott(bow_train, bow_test, lott_train, lott_test,
                  y_train, y_test, alpha=0.5, k1=1.5, b=0.75,
                  n_neighbors=7):
    from knn_classifier import predict

    # fit BM25 once on training set
    bm25 = BM25(k1=k1, b=b).fit(bow_train)
    
    # precompute ALL bm25 scores at once — (n_test, n_train)
    all_bm25_scores = bm25.score_matrix(bow_test)
    
    # precompute ALL lott distances at once — (n_test, n_train)
    # vectorised: broadcast subtraction
    lott_train_arr = np.array(lott_train)
    lott_test_arr  = np.array(lott_test)
    # (n_test, n_train)
    all_lott_dists = cdist(lott_test_arr, lott_train_arr, metric='euclidean')

    n_classes = len(np.unique(y_train))
    predictions = []

    for i in range(len(bow_test)):
        combined = combine_bm25_lott(
            all_bm25_scores[i],
            all_lott_dists[i],
            alpha=alpha
        )
        rank = np.argsort(combined)[:n_neighbors]
        predictions.append(predict(y_train[rank], n_classes))

    test_error = 1 - (np.array(predictions) == y_test).mean()
    return test_error