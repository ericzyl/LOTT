# Main Pipeline for complete RAG System
import numpy as np
import pickle
from pathlib import Path
import config
from dataset_loader import TRECCovidLoader
from preprocessing import TextPreprocessor, prepare_bow_data
from lda_trainer import train_and_save_lda, LDATrainer
from bert_embeddings import generate_and_save_bert_embeddings
from lott_embeddings import generate_and_save_lott_embeddings


def check_file_exists(filepath):
    return Path(filepath).exists()


def step_1_load_dataset():
    # Loading and Saving TREC-COVID Dataset
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    if check_file_exists(config.DOCUMENTS_TRAIN_PATH):
        print("Dataset already loaded. Skipping...")
        train_corpus, test_corpus, queries, qrels = TRECCovidLoader.load_saved_data()
    else:
        loader = TRECCovidLoader(max_docs=config.MAX_DOCS)
        corpus, queries, qrels = loader.load_dataset()
        train_corpus, test_corpus = loader.split_corpus(random_seed=config.RANDOM_SEED)
        loader.save_data(train_corpus, test_corpus)
    
    return train_corpus, test_corpus, queries, qrels


def step_2_preprocess_data(train_corpus, test_corpus):
    # Building Vocabulary and Creating BoW Representations
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)
    
    if check_file_exists(config.VOCAB_PATH) and check_file_exists(config.WORD_EMBEDDINGS_PATH):
        print("Vocabulary and embeddings already prepared. Loading...")
        
        with open(config.VOCAB_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        vocab = vocab_data['vocab']
        
        embeddings = np.load(config.WORD_EMBEDDINGS_PATH)
        
        # Recreating BoW Data
        preprocessor = TextPreprocessor()
        preprocessor.load_vocabulary()
        
        bow_train, train_doc_ids = preprocessor.corpus_to_bow(train_corpus)
        bow_test, test_doc_ids = preprocessor.corpus_to_bow(test_corpus)
        
        print(f"Loaded vocabulary: {len(vocab)} words")
        print(f"BoW train shape: {bow_train.shape}")
        print(f"BoW test shape: {bow_test.shape}")
    else:
        bow_train, bow_test, vocab, embeddings, train_doc_ids, test_doc_ids = prepare_bow_data(
            train_corpus, test_corpus
        )
    
    return bow_train, bow_test, vocab, embeddings, train_doc_ids, test_doc_ids


def step_3_train_lda(bow_train, embeddings, vocab):
    # Step 3: Training LDA model and computing topic costs
    print("\n" + "="*80)
    print("STEP 3: TRAINING LDA MODEL")
    print("="*80)
    
    if check_file_exists(config.LDA_MODEL_PATH):
        print("LDA model already trained. Loading...")
        lda_model, topics, lda_centers, topic_cost_matrix = LDATrainer.load_artifacts()
    else:
        trainer = train_and_save_lda(bow_train, embeddings, vocab)
        lda_model = trainer.model
        topics = trainer.topics
        lda_centers = trainer.lda_centers
        topic_cost_matrix = trainer.topic_cost_matrix
    
    return lda_model, topics, lda_centers, topic_cost_matrix


def step_4_generate_bert_embeddings(train_corpus, test_corpus, train_doc_ids, test_doc_ids):
    # Step 4: Generating BERT embeddings
    print("\n" + "="*80)
    print("STEP 4: GENERATING BERT EMBEDDINGS")
    print("="*80)
    
    if check_file_exists(config.BERT_TRAIN_EMBEDDINGS) and check_file_exists(config.BERT_TEST_EMBEDDINGS):
        print("BERT embeddings already generated. Loading...")
        from bert_embeddings import load_bert_embeddings
        train_embeddings, test_embeddings = load_bert_embeddings()
    else:
        train_embeddings, test_embeddings = generate_and_save_bert_embeddings(
            train_corpus, test_corpus, train_doc_ids, test_doc_ids
        )
    
    return train_embeddings, test_embeddings


# def step_5_generate_lott_embeddings(bow_train, bow_test):
#     # Step 5: Generating LOTT embeddings
#     print("\n" + "="*80)
#     print("STEP 5: GENERATING LOTT EMBEDDINGS")
#     print("="*80)
    
#     if check_file_exists(config.LOTT_TRAIN_EMBEDDINGS) and check_file_exists(config.LOTT_TEST_EMBEDDINGS):
#         print("LOTT embeddings already generated. Loading...")
#         from lott_embeddings import load_lott_embeddings
#         train_embeddings, test_embeddings, train_topics, test_topics = load_lott_embeddings()
#     else:
#         train_embeddings, test_embeddings, train_topics, test_topics = generate_and_save_lott_embeddings(
#             bow_train, bow_test
#         )
    
#     return train_embeddings, test_embeddings, train_topics, test_topics

def step_5_generate_lott_embeddings(bow_all):
    print("\n" + "="*80)
    print("STEP 5: GENERATING LOTT EMBEDDINGS")
    print("="*80)
    
    if check_file_exists(config.LOTT_TRAIN_EMBEDDINGS):
        print("LOTT embeddings already generated. Loading...")
        from lott_embeddings import load_lott_embeddings
        all_embeddings = np.load(config.LOTT_TRAIN_EMBEDDINGS)
        all_topics = np.load(config.TOPIC_PROPORTIONS_TRAIN)
    else:
        all_embeddings, all_topics = generate_and_save_lott_embeddings(bow_all)
    
    return all_embeddings, all_topics


def save_pipeline_metadata(train_doc_ids, test_doc_ids):
    # Saving Metadata about Document ordering
    metadata = {
        'train_doc_ids': train_doc_ids,
        'test_doc_ids': test_doc_ids
    }
    
    metadata_path = config.DATA_DIR / 'pipeline_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nPipeline metadata saved to {metadata_path}")


def run_full_pipeline():
    print("\n" + "="*80)
    print("STARTING FULL RAG PIPELINE")
    print("="*80)
    
    config.print_config()
    
    train_corpus, test_corpus, queries, qrels = step_1_load_dataset()
    
    bow_train, bow_test, vocab, embeddings, train_doc_ids, test_doc_ids = step_2_preprocess_data(
        train_corpus, test_corpus
    )
    
    lda_model, topics, lda_centers, topic_cost_matrix = step_3_train_lda(
        bow_train, embeddings, vocab
    )
    
    bert_train_emb, bert_test_emb = step_4_generate_bert_embeddings(
        train_corpus, test_corpus, train_doc_ids, test_doc_ids
    )
    
    lott_train_emb, lott_test_emb, train_topics, test_topics = step_5_generate_lott_embeddings(
        bow_train, bow_test
    )
    
    save_pipeline_metadata(train_doc_ids, test_doc_ids)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated artifacts:")
    print(f"  - Train documents: {len(train_doc_ids)}")
    print(f"  - Test documents: {len(test_doc_ids)}")
    print(f"  - Queries: {len(queries)}")
    print(f"  - BERT train embeddings: {bert_train_emb.shape}")
    print(f"  - BERT test embeddings: {bert_test_emb.shape}")
    print(f"  - LOTT train embeddings: {lott_train_emb.shape}")
    print(f"  - LOTT test embeddings: {lott_test_emb.shape}")
    print("\nAll embeddings are ready for RAG evaluation!")
    
    return {
        'train_corpus': train_corpus,
        'test_corpus': test_corpus,
        'queries': queries,
        'qrels': qrels,
        'train_doc_ids': train_doc_ids,
        'test_doc_ids': test_doc_ids,
        'bert_train_emb': bert_train_emb,
        'bert_test_emb': bert_test_emb,
        'lott_train_emb': lott_train_emb,
        'lott_test_emb': lott_test_emb
    }


if __name__ == "__main__":
    pipeline_data = run_full_pipeline()
    
    print("\n" + "="*80)
    print("READY FOR RAG SYSTEM EVALUATION")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run rag_bert.py to build BERT-based RAG system")
    print("  2. Run rag_lott.py to build LOTT-based RAG system")
    print("  3. Run evaluate_rag.py to compare both systems")