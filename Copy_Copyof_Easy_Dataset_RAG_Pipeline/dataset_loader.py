import pickle
from typing import Dict, List, Tuple
import numpy as np
import config
from beir import util
from beir.datasets.data_loader import GenericDataLoader

class MSMARCOLoader:

    def __init__(self, max_docs=None):
        self.max_docs = max_docs
        self.corpus = None
        self.queries = None
        self.qrels = None
        
    def load_dataset(self):
        print("Loading MS MARCO dataset from BeIR...")
        
        # Download and load using BeIR
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        data_path = util.download_and_unzip(url, "datasets")
        
        # Load the dataset
        # MS MARCO: 8.8M passages, 6,980 dev queries (we use dev for evaluation)
        beir_corpus, beir_queries, beir_qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
        
        print(f"Total documents in corpus: {len(beir_corpus)}")
        print(f"Total queries: {len(beir_queries)}")
        print(f"Total query-document pairs in qrels: {sum(len(docs) for docs in beir_qrels.values())}")
        
        # Process corpus
        corpus_items = list(beir_corpus.items())
        if self.max_docs:
            corpus_items = corpus_items[:self.max_docs]
            print(f"Limited to {len(corpus_items)} documents for testing")
        
        self.corpus = {}
        print("Processing corpus...")
        for idx, (doc_id, doc_data) in enumerate(corpus_items):
            if idx % 50000 == 0:
                print(f"  Processed {idx}/{len(corpus_items)} documents")
            
            title = doc_data.get('title', '')
            text = doc_data.get('text', '')
            
            # MS MARCO doesn't use titles much, mostly just passages
            full_text = f"{title}. {text}" if title else text
            
            self.corpus[doc_id] = {
                'text': full_text,
                'title': title,
                'doc_id': doc_id
            }
        
        print(f"Loaded {len(self.corpus)} documents")
        
        # Use BeIR queries and qrels directly
        self.queries = beir_queries
        self.qrels = beir_qrels
        
        print(f"Loaded {len(self.queries)} queries")
        print(f"Loaded qrels for {len(self.qrels)} queries")
        
        # Print statistics about relevant docs per query
        relevant_counts = [len(docs) for docs in self.qrels.values()]
        print(f"\nRelevant docs per query - Avg: {sum(relevant_counts)/len(relevant_counts):.1f}, "
              f"Min: {min(relevant_counts)}, Max: {max(relevant_counts)}")
        
        return self.corpus, self.queries, self.qrels
    
    def save_data(self):
        print("Saving processed data...")
        
        with open(config.DOCUMENTS_PATH, 'wb') as f:
            pickle.dump(self.corpus, f)
        
        with open(config.QUERIES_PATH, 'wb') as f:
            pickle.dump(self.queries, f)
        
        with open(config.QRELS_PATH, 'wb') as f:
            pickle.dump(self.qrels, f)
        
        print("Data saved successfully!")
    
    @staticmethod
    def load_saved_data():
        print("Loading saved data...")
        
        with open(config.DOCUMENTS_PATH, 'rb') as f:
            corpus = pickle.load(f)
        
        with open(config.QUERIES_PATH, 'rb') as f:
            queries = pickle.load(f)
        
        with open(config.QRELS_PATH, 'rb') as f:
            qrels = pickle.load(f)
        
        print(f"Loaded: {len(corpus)} documents")
        print(f"Loaded: {len(queries)} queries, {len(qrels)} query-relevance mappings")
        
        return corpus, queries, qrels


def main():
    loader = MSMARCOLoader(max_docs=config.MAX_DOCS)
    
    # Load dataset
    corpus, queries, qrels = loader.load_dataset()
    
    # Save data (NO SPLIT - use all documents for retrieval)
    loader.save_data()
    
    print("\nDataset loading complete!")
    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)}")
    print(f"Relevance judgments: {len(qrels)}")


if __name__ == "__main__":
    main()