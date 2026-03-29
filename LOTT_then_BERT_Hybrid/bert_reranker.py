"""
BERT Reranker  –  second stage of the LOTT → BERT hybrid pipeline.

Given a *candidate pool* per query (from LOTT retrieval), computes BERT
cosine similarities and returns the top-K documents.

Also used as the *standalone* BERT-only baseline for fair comparison.
"""

from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


class BERTReranker:
    """
    Encodes documents and queries with SBERT.
    Can be used in two modes:
      1. Reranker  – scores only the LOTT candidate subset per query.
      2. Full FAISS baseline – builds an index over all docs and retrieves top-K.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name   = dataset_name
        self.cache          = config.get_cache_paths(dataset_name)
        self.model: SentenceTransformer = None
        self.doc_embeddings: np.ndarray  = None
        self.doc_ids:        List[str]   = None
        # Full-index components (used for BERT-only baseline)
        self._faiss_index:   faiss.Index = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.model is None:
            print(f"Loading SBERT model: {config.BERT_MODEL}...")
            self.model = SentenceTransformer(config.BERT_MODEL)
            print(f"Model loaded (dim={self.model.get_sentence_embedding_dimension()})")

    # ------------------------------------------------------------------
    # Document embeddings
    # ------------------------------------------------------------------

    def prepare_documents(self, corpus: Dict, doc_ids: List[str]):
        """Compute (or load cached) BERT doc embeddings."""
        self.doc_ids = doc_ids

        if self.cache['bert_embeddings'].exists():
            print("Loading BERT doc embeddings from cache...")
            self.doc_embeddings = np.load(self.cache['bert_embeddings'])
            return

        self._load_model()
        texts = [corpus[d]['text'] for d in doc_ids]
        print(f"Encoding {len(texts)} documents with BERT...")
        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=config.BERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(self.cache['bert_embeddings'], self.doc_embeddings)
        print(f"BERT doc embeddings cached → {self.cache['bert_embeddings']}")

    # ------------------------------------------------------------------
    # Query embeddings
    # ------------------------------------------------------------------

    def encode_queries(self, queries: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        query_ids = list(queries.keys())

        if self.cache['bert_query_embeddings'].exists():
            print("Loading BERT query embeddings from cache...")
            q_embs = np.load(self.cache['bert_query_embeddings'])
            return q_embs, query_ids

        self._load_model()
        texts = [queries[qid] for qid in query_ids]
        print(f"Encoding {len(texts)} queries with BERT...")
        q_embs = self.model.encode(
            texts,
            batch_size=config.BERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(self.cache['bert_query_embeddings'], q_embs)
        return q_embs, query_ids

    # ------------------------------------------------------------------
    # Mode 1: Reranker over LOTT candidate pool
    # ------------------------------------------------------------------

    def rerank(
        self,
        query_embeddings:     np.ndarray,
        lott_candidate_ids:   List[List[str]],
        k:                    int = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        For each query, score only its LOTT candidate documents with BERT
        cosine similarity, then return the top-k.
        """
        if k is None:
            k = config.K_FINAL

        if self.doc_embeddings is None:
            raise RuntimeError("Call prepare_documents() before rerank().")

        doc_id_to_idx = {d: i for i, d in enumerate(self.doc_ids)}

        # Normalise query embeddings once
        q_norm = query_embeddings.astype('float32').copy()
        faiss.normalize_L2(q_norm)

        all_doc_ids = []
        all_scores  = []

        print(f"BERT reranking {len(query_embeddings)} queries "
              f"(pool={len(lott_candidate_ids[0])}, final k={k})...")

        for qi, (q_emb, candidates) in enumerate(zip(q_norm, lott_candidate_ids)):
            if qi % 500 == 0:
                print(f"  Reranking query {qi}/{len(query_embeddings)}")

            # Gather candidate embeddings
            idxs      = [doc_id_to_idx[d] for d in candidates if d in doc_id_to_idx]
            valid_ids = [candidates[i] for i, d in enumerate(candidates) if d in doc_id_to_idx]

            if not idxs:
                all_doc_ids.append([])
                all_scores.append([])
                continue

            cand_embs = self.doc_embeddings[idxs].astype('float32')
            faiss.normalize_L2(cand_embs)

            scores = cand_embs @ q_emb  # cosine similarity
            top_k  = min(k, len(scores))
            order  = np.argsort(-scores)[:top_k]

            all_doc_ids.append([valid_ids[i] for i in order])
            all_scores.append([float(scores[i]) for i in order])

        return all_doc_ids, all_scores

    # ------------------------------------------------------------------
    # Mode 2: Standalone BERT-only baseline (full FAISS index)
    # ------------------------------------------------------------------

    def build_full_index(self):
        """Build a FAISS flat-IP index over all document embeddings."""
        if self.doc_embeddings is None:
            raise RuntimeError("Call prepare_documents() before build_full_index().")

        print(f"Building full BERT FAISS index ({len(self.doc_ids)} docs)...")
        embs_f32 = self.doc_embeddings.astype('float32').copy()
        faiss.normalize_L2(embs_f32)
        self._faiss_index = faiss.IndexFlatIP(embs_f32.shape[1])
        self._faiss_index.add(embs_f32)
        print(f"BERT FAISS index ready – {self._faiss_index.ntotal} vectors.")

    def retrieve_full(
        self,
        query_embeddings: np.ndarray,
        k: int = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Pure BERT retrieval over the full corpus (baseline)."""
        if k is None:
            k = config.K_FINAL
        if self._faiss_index is None:
            self.build_full_index()

        q_f32 = query_embeddings.astype('float32').copy()
        faiss.normalize_L2(q_f32)

        print(f"BERT-only retrieval: top-{k} for {len(q_f32)} queries...")
        sims, idxs = self._faiss_index.search(q_f32, k)

        all_doc_ids = [[self.doc_ids[i] for i in row] for row in idxs]
        all_scores  = [row.tolist() for row in sims]
        return all_doc_ids, all_scores