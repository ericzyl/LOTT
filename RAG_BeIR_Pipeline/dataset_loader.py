# Dataset Loader for TREC-COVID Dataset using BeIR package
import pickle
from datasets import load_dataset
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import config
from beir import util
from beir.datasets.data_loader import GenericDataLoader

class TRECCovidLoader:
    def __init__(self, max_docs=None):
        self.max_docs = max_docs
        self.corpus = None
        self.queries = None
        self.qrels = None
        
    def load_dataset(self):
        print("Loading TREC-COVID dataset from BeIR...")
        
        # dataset = load_dataset("BeIR/trec-covid", "corpus")
        # Downloading and Loading using BeIR
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
        data_path = util.download_and_unzip(url, "datasets")
        beir_corpus, beir_queries, beir_qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        # # Getting Corpus (documents)
        # corpus_data = dataset['corpus']
        
        # print(f"Total documents in corpus: {len(corpus_data)}")
        
        # # Limiting Documents if specified
        # if self.max_docs:
        #     corpus_data = corpus_data.select(range(min(self.max_docs, len(corpus_data))))
        #     print(f"Limited to {len(corpus_data)} documents for testing")
        
        # # Extracting Documents with metadata
        # self.corpus = {}
        # for item in tqdm(corpus_data, desc="Processing corpus"):
        #     doc_id = item['_id']
        #     title = item.get('title', '')
        #     text = item.get('text', '')
            
        #     # Combining title and text
        #     full_text = f"{title}. {text}" if title else text
            
        #     self.corpus[doc_id] = {
        #         'text': full_text,
        #         'title': title,
        #         'doc_id': doc_id
        #     }
        print(f"Total documents in corpus: {len(beir_corpus)}")

        corpus_items = list(beir_corpus.items())
        if self.max_docs:
            corpus_items = corpus_items[:self.max_docs]
            print(f"Limited to {len(corpus_items)} documents for testing")

        self.corpus = {}
        for doc_id, doc_data in tqdm(corpus_items, desc="Processing corpus"):
            title = doc_data.get('title', '')
            text = doc_data.get('text', '')
            full_text = f"{title}. {text}" if title else text
            
            self.corpus[doc_id] = {
                'text': full_text,
                'title': title,
                'doc_id': doc_id
            }
        
        print(f"Loaded {len(self.corpus)} documents")
        
        # Loading Queries
        try:
            # queries_dataset = load_dataset("BeIR/trec-covid", "queries")
            # queries_data = queries_dataset['queries']
            
            # self.queries = {}
            # for item in queries_data:
            #     query_id = item['_id']
            #     query_text = item['text']
            #     self.queries[query_id] = query_text
            # Use BeIR queries directly
            self.queries = beir_queries

            print(f"Loaded {len(self.queries)} queries")
        except Exception as e:
            print(f"Could not load queries: {e}")
            self.queries = {}
        
        # Loading relevance Judgments (qrels)
        try:
            # qrels_dataset = load_dataset("BeIR/trec-covid", "qrels")
            
            # # Combining all splits if available
            # all_qrels = []
            # for split in qrels_dataset.keys():
            #     all_qrels.extend(qrels_dataset[split])
            
            # # Organizing qrels: {query_id: {doc_id: relevance_score}}
            # self.qrels = {}
            # for item in all_qrels:
            #     query_id = item['query-id']
            #     doc_id = item['corpus-id']
            #     score = item['score']
                
            #     if query_id not in self.qrels:
            #         self.qrels[query_id] = {}
            #     self.qrels[query_id][doc_id] = score

            # Using BeIR qrels directly
            self.qrels = beir_qrels
            
            print(f"Loaded qrels for {len(self.qrels)} queries")
        except Exception as e:
            print(f"Could not load qrels: {e}")
            self.qrels = {}
        
        return self.corpus, self.queries, self.qrels
    
    def split_corpus(self, train_ratio=0.75, random_seed=42):
        # Splitting Corpus into Train and Test sets
        np.random.seed(random_seed)
        
        doc_ids = list(self.corpus.keys())
        n_train = int(len(doc_ids) * train_ratio)
        
        # Shuffling Document IDs
        shuffled_ids = np.random.permutation(doc_ids)
        
        train_ids = shuffled_ids[:n_train]
        test_ids = shuffled_ids[n_train:]
        
        train_corpus = {doc_id: self.corpus[doc_id] for doc_id in train_ids}
        test_corpus = {doc_id: self.corpus[doc_id] for doc_id in test_ids}
        
        print(f"Split corpus: {len(train_corpus)} train, {len(test_corpus)} test")
        
        return train_corpus, test_corpus
    
    def save_data(self, train_corpus, test_corpus):
        # Saving Processed Data to Disk
        print("Saving processed data...")
        
        with open(config.DOCUMENTS_TRAIN_PATH, 'wb') as f:
            pickle.dump(train_corpus, f)
        
        with open(config.DOCUMENTS_TEST_PATH, 'wb') as f:
            pickle.dump(test_corpus, f)
        
        with open(config.QUERIES_PATH, 'wb') as f:
            pickle.dump(self.queries, f)
        
        with open(config.QRELS_PATH, 'wb') as f:
            pickle.dump(self.qrels, f)
        
        print("Data saved successfully!")
    
    @staticmethod
    def load_saved_data():
        # Loading saved Data
        print("Loading saved data...")
        
        with open(config.DOCUMENTS_TRAIN_PATH, 'rb') as f:
            train_corpus = pickle.load(f)
        
        with open(config.DOCUMENTS_TEST_PATH, 'rb') as f:
            test_corpus = pickle.load(f)
        
        with open(config.QUERIES_PATH, 'rb') as f:
            queries = pickle.load(f)
        
        with open(config.QRELS_PATH, 'rb') as f:
            qrels = pickle.load(f)
        
        print(f"Loaded: {len(train_corpus)} train docs, {len(test_corpus)} test docs")
        print(f"Loaded: {len(queries)} queries, {len(qrels)} query-relevance mappings")
        
        return train_corpus, test_corpus, queries, qrels


def main():
    loader = TRECCovidLoader(max_docs=config.MAX_DOCS)
    
    corpus, queries, qrels = loader.load_dataset()
    
    train_corpus, test_corpus = loader.split_corpus(random_seed=config.RANDOM_SEED)
    
    loader.save_data(train_corpus, test_corpus)
    
    print("\nDataset loading complete!")
    print(f"Train corpus: {len(train_corpus)} documents")
    print(f"Test corpus: {len(test_corpus)} documents")
    print(f"Queries: {len(queries)}")
    print(f"Relevance judgments: {len(qrels)}")


if __name__ == "__main__":
    main()