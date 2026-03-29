import pickle
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances

import config


def _sparse_ot(w1: np.ndarray, w2: np.ndarray, M: np.ndarray) -> float:
    import ot
    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()
    a1 = np.where(w1)[0];  a2 = np.where(w2)[0]
    M_red = np.ascontiguousarray(M[a1][:, a2])
    return ot.emd2(w1[a1], w2[a2], M_red)


class LDAModule:

    def __init__(self, dataset_name: str):
        self.dataset_name     = dataset_name
        self.cache            = config.get_cache_paths(dataset_name)
        self.model            = None
        self.topics           = None
        self.lda_centers      = None
        self.topic_cost_matrix = None

    # ------------------------------------------------------------------
    def train_or_load(
        self,
        bow_data:   np.ndarray,
        embeddings: np.ndarray,
        vocab:      List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self._load_from_cache():
            return self.topics, self.lda_centers, self.topic_cost_matrix

        print(f"Training LDA with {config.N_TOPICS} topics "
              f"({config.LDA_MAX_ITER} iterations)...")
        self.model = LatentDirichletAllocation(
            n_components=config.N_TOPICS,
            max_iter=config.LDA_MAX_ITER,
            random_state=config.LDA_RANDOM_STATE,
            evaluate_every=10,
            perp_tol=1e-3,
            n_jobs=1,
            verbose=1,
        )
        self.model.fit(bow_data)

        # Normalise topic-word distributions
        self.topics = self.model.components_.copy()
        self.topics = self.topics / self.topics.sum(axis=1, keepdims=True)

        # Sparsify: keep top-N words per topic
        print(f"Sparsifying topics (top-{config.N_TOP_WORDS} words)...")
        for k in range(config.N_TOPICS):
            zero_idx = np.argsort(-self.topics[k])[config.N_TOP_WORDS:]
            self.topics[k][zero_idx] = 0.0
            s = self.topics[k].sum()
            if s > 0:
                self.topics[k] /= s

        # Topic centres in embedding space
        self.lda_centers = np.matmul(self.topics, embeddings)
        print(f"LDA centres shape: {self.lda_centers.shape}")

        # Topic-topic cost matrix via sparse OT
        print("Computing topic–topic cost matrix (upper triangle)...")
        cost_emb = euclidean_distances(embeddings, embeddings)
        T = config.N_TOPICS
        self.topic_cost_matrix = np.zeros((T, T))
        for i in range(T):
            if i % 10 == 0:
                print(f"  Topic cost progress: {i}/{T}")
            for j in range(i + 1, T):
                self.topic_cost_matrix[i, j] = _sparse_ot(
                    self.topics[i], self.topics[j], cost_emb
                )
        self.topic_cost_matrix += self.topic_cost_matrix.T

        self._save_to_cache()
        print("LDA training complete.")
        return self.topics, self.lda_centers, self.topic_cost_matrix

    # ------------------------------------------------------------------
    def infer_topics(self, bow_data: np.ndarray, cache_key: str = 'topic_proportions') -> np.ndarray:
        cache_path = self.cache[cache_key]
        if cache_path.exists():
            print(f"Loading topic proportions from cache ({cache_key})...")
            return np.load(cache_path)

        print(f"Inferring topic proportions for {len(bow_data)} items...")
        props = self.model.transform(bow_data)
        np.save(cache_path, props)
        return props

    # ------------------------------------------------------------------
    def _load_from_cache(self) -> bool:
        if not self.cache['lda_model'].exists():
            return False
        print("Loading LDA from cache...")
        with open(self.cache['lda_model'], 'rb') as f:
            self.model = pickle.load(f)
        self.topics            = np.load(self.cache['lda_topics'])
        self.lda_centers       = np.load(self.cache['lda_centers'])
        self.topic_cost_matrix = np.load(self.cache['topic_cost_matrix'])
        print(f"LDA loaded – topics {self.topics.shape}, "
              f"centres {self.lda_centers.shape}, "
              f"cost {self.topic_cost_matrix.shape}")
        return True

    def _save_to_cache(self):
        print("Caching LDA artefacts...")
        with open(self.cache['lda_model'], 'wb') as f:
            pickle.dump(self.model, f)
        np.save(self.cache['lda_topics'],        self.topics)
        np.save(self.cache['lda_centers'],       self.lda_centers)
        np.save(self.cache['topic_cost_matrix'], self.topic_cost_matrix)
        print("LDA artefacts cached.")


if __name__ == "__main__":
    from dataset_loader import load_dataset
    from preprocessing  import prepare_bow_data

    corpus, _, _ = load_dataset("msmarco")
    bow_data, vocab, embeddings, _ = prepare_bow_data(corpus, "msmarco")

    lda = LDAModule("msmarco")
    topics, centres, costs = lda.train_or_load(bow_data, embeddings, vocab)
    print(f"Topics shape : {topics.shape}")
    print(f"Centres shape: {centres.shape}")
    print(f"Cost shape   : {costs.shape}")