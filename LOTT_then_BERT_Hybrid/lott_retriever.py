"""
LOTT Retriever  –  first stage of the LOTT → BERT hybrid pipeline.

Builds LOTT (Linearised Optimal Transport Topics) embeddings for every
document, indexes them with FAISS, and retrieves a *broad* candidate set
(K_LOTT_RETRIEVAL documents) for each query.  BERT then re-ranks that pool.
"""

from typing import Dict, List, Tuple

import faiss
import numpy as np
import ot

import config


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_gaussian_1d(size: int, fwhm: float = 25.0) -> np.ndarray:
    centre = size // 2
    x = np.arange(size)
    g = np.exp(-4 * np.log(2) * ((x - centre) ** 2) / fwhm ** 2)
    return g / g.sum()


def _lot_embedding(coupling: np.ndarray, lda_centers: np.ndarray) -> np.ndarray:
    """Barycentric projection of one row of the OT coupling matrix."""
    result = []
    for i in range(coupling.shape[0]):
        w = coupling[i].sum()
        if w == 0:
            result.append(np.zeros(lda_centers.shape[1]))
        else:
            nz  = np.where(coupling[i] > 0)[0]
            yc  = (coupling[i][nz, None] * lda_centers[nz]).sum(axis=0) / w
            result.append(yc)
    return np.array(result).flatten()


def _create_lott_embeddings(
    topic_proportions: np.ndarray,
    reference_dist:    np.ndarray,
    lda_centers:       np.ndarray,
    topic_cost_matrix: np.ndarray,
    label:             str = "items",
) -> np.ndarray:
    # print(f"Generating LOTT embeddings for {len(topic_proportions)} {label}...")
    n = topic_proportions.shape[0]
    print(f"Generating LOTT embeddings for {n} {label}...")
    embeddings = []
    # for i, props in enumerate(topic_proportions):
    #     if i % 2_000 == 0:
    #         print(f"  LOTT progress: {i}/{len(topic_proportions)}")
    for i, props in enumerate(topic_proportions):
        if i % 2_000 == 0:
            print(f"  LOTT progress: {i}/{n}")
        props = props / props.sum() if props.sum() > 0 else np.ones_like(props) / len(props)
        coupling = ot.emd(props, reference_dist, topic_cost_matrix)
        embeddings.append(_lot_embedding(coupling, lda_centers))
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LOTTRetriever:
    """
    Builds and searches a FAISS index over LOTT document embeddings.
    Used as the *first* (broad) retrieval stage.
    """

    def __init__(self, dataset_name: str, lda_module):
        self.dataset_name = dataset_name
        self.cache        = config.get_cache_paths(dataset_name)
        self.lda_module   = lda_module
        self.ref_dist     = _make_gaussian_1d(config.N_TOPICS, fwhm=config.GAUSSIAN_FWHM)

        self.doc_ids:      List[str]    = None
        self.doc_lott_embs: np.ndarray  = None
        self.index:         faiss.Index = None
        self.emb_dim:       int         = None

    # ------------------------------------------------------------------
    # Document index
    # ------------------------------------------------------------------

    def build_index(self, bow_data: np.ndarray, doc_ids: List[str]):
        """Compute (or load cached) LOTT doc embeddings and build FAISS index."""
        self.doc_ids = doc_ids

        # ---- document LOTT embeddings ----
        if self.cache['lott_embeddings'].exists():
            print("Loading LOTT doc embeddings from cache...")
            self.doc_lott_embs = np.load(self.cache['lott_embeddings'])
        else:
            doc_topic_props = self.lda_module.infer_topics(bow_data, 'topic_proportions')
            self.doc_lott_embs = _create_lott_embeddings(
                doc_topic_props,
                self.ref_dist,
                self.lda_module.lda_centers,
                self.lda_module.topic_cost_matrix,
                label="documents",
            )
            np.save(self.cache['lott_embeddings'], self.doc_lott_embs)
            print(f"LOTT doc embeddings saved → {self.cache['lott_embeddings']}")

        self.emb_dim = self.doc_lott_embs.shape[1]
        print(f"LOTT embedding dim: {self.emb_dim}")

        # ---- FAISS index ----
        if self.cache['faiss_lott_index'].exists():
            print("Loading FAISS LOTT index from cache...")
            self.index = faiss.read_index(str(self.cache['faiss_lott_index']))
        else:
            print(f"Building FAISS index over {len(doc_ids)} LOTT vectors...")
            embs_f32 = self.doc_lott_embs.astype('float32')
            faiss.normalize_L2(embs_f32)
            self.index = faiss.IndexFlatIP(self.emb_dim)
            self.index.add(embs_f32)
            faiss.write_index(self.index, str(self.cache['faiss_lott_index']))
            print(f"FAISS LOTT index saved → {self.cache['faiss_lott_index']}")

        print(f"LOTT retriever ready – {self.index.ntotal} docs indexed.")

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def encode_queries(self, queries: Dict[str, str], vocab_data: dict) -> Tuple[np.ndarray, List[str]]:
        """Produce LOTT embeddings for all queries."""
        from preprocessing import TextPreprocessor

        query_ids = list(queries.keys())

        if self.cache['lott_query_embeddings'].exists():
            print("Loading LOTT query embeddings from cache...")
            q_embs = np.load(self.cache['lott_query_embeddings'])
            return q_embs, query_ids

        # BoW for queries
        prep = TextPreprocessor()
        prep.vocab       = vocab_data['vocab']
        prep.word_to_idx = vocab_data['word_to_idx']
        query_bow = np.array([prep.text_to_bow(queries[qid]) for qid in query_ids])

        # Topic proportions
        q_topic_props = self.lda_module.infer_topics(query_bow, 'query_topic_proportions')

        # LOTT embeddings
        q_embs = _create_lott_embeddings(
            q_topic_props,
            self.ref_dist,
            self.lda_module.lda_centers,
            self.lda_module.topic_cost_matrix,
            label="queries",
        )
        np.save(self.cache['lott_query_embeddings'], q_embs)
        return q_embs, query_ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embeddings: np.ndarray,
        k: int = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        if k is None:
            k = config.K_LOTT_RETRIEVAL

        print(f"LOTT retrieval: top-{k} for {len(query_embeddings)} queries...")
        q_f32 = query_embeddings.astype('float32')
        faiss.normalize_L2(q_f32)

        similarities, indices = self.index.search(q_f32, k)

        all_doc_ids = []
        all_scores  = []
        for i in range(len(query_embeddings)):
            all_doc_ids.append([self.doc_ids[idx] for idx in indices[i]])
            all_scores.append(similarities[i].tolist())

        return all_doc_ids, all_scores