"""
Main pipeline to orchestrate the entire RAG system setup
Runs all preprocessing, training, and embedding generation steps
"""
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
    """Check if a file exists"""
    return Path(filepath).exists()


def step_1_load_dataset():
    """Step 1: Load and save dataset (NO SPLIT)"""
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    if check_file_exists(config.DOCUMENTS_PATH):
        print("Dataset already loaded. Skipping...")
        corpus, queries, qrels = TRECCovidLoader.load_saved_data()
    else:
        loader = TRECCovidLoader(max_docs=config.MAX_DOCS)
        corpus, queries, qrels = loader.load_dataset()
        loader.save_data()
    
    return corpus, queries, qrels


def step_2_preprocess_data(corpus):
    """Step 2: Build vocabulary and create BoW representations"""
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)
    
    if check_file_exists(config.VOCAB_PATH) and check_file_exists(config.WORD_EMBEDDINGS_PATH):
        print("Vocabulary and embeddings already prepared. Loading...")
        
        # Load vocabulary
        with open(config.VOCAB_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        vocab = vocab_data['vocab']
        
        # Load word embeddings
        embeddings = np.load(config.WORD_EMBEDDINGS_PATH)
        
        # Recreate BoW data
        preprocessor = TextPreprocessor()
        preprocessor.load_vocabulary()
        
        bow_data, doc_ids = preprocessor.corpus_to_bow(corpus)
        
        print(f"Loaded vocabulary: {len(vocab)} words")
        print(f"BoW shape: {bow_data.shape}")
    else:
        bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus)
    
    return bow_data, vocab, embeddings, doc_ids


def step_3_train_lda(bow_data, embeddings, vocab):
    """Step 3: Train LDA model and compute topic costs"""
    print("\n" + "="*80)
    print("STEP 3: TRAINING LDA MODEL")
    print("="*80)
    
    if check_file_exists(config.LDA_MODEL_PATH):
        print("LDA model already trained. Loading...")
        lda_model, topics, lda_centers, topic_cost_matrix = LDATrainer.load_artifacts()
    else:
        trainer = train_and_save_lda(bow_data, embeddings, vocab)
        lda_model = trainer.model
        topics = trainer.topics
        lda_centers = trainer.lda_centers
        topic_cost_matrix = trainer.topic_cost_matrix
    
    return lda_model, topics, lda_centers, topic_cost_matrix


def step_4_generate_bert_embeddings(corpus, doc_ids):
    """Step 4: Generate BERT embeddings"""
    print("\n" + "="*80)
    print("STEP 4: GENERATING BERT EMBEDDINGS")
    print("="*80)
    
    if check_file_exists(config.BERT_EMBEDDINGS):
        print("BERT embeddings already generated. Loading...")
        from bert_embeddings import load_bert_embeddings
        embeddings = load_bert_embeddings()
    else:
        embeddings = generate_and_save_bert_embeddings(corpus, doc_ids)
    
    return embeddings


def step_5_generate_lott_embeddings(bow_data):
    """Step 5: Generate LOTT embeddings"""
    print("\n" + "="*80)
    print("STEP 5: GENERATING LOTT EMBEDDINGS")
    print("="*80)
    
    if check_file_exists(config.LOTT_EMBEDDINGS):
        print("LOTT embeddings already generated. Loading...")
        from lott_embeddings import load_lott_embeddings
        embeddings, topics = load_lott_embeddings()
    else:
        embeddings, topics = generate_and_save_lott_embeddings(bow_data)
    
    return embeddings, topics


def save_pipeline_metadata(doc_ids):
    """Save metadata about document ordering"""
    metadata = {
        'doc_ids': doc_ids
    }
    
    metadata_path = config.DATA_DIR / 'pipeline_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nPipeline metadata saved to {metadata_path}")


def run_full_pipeline():
    """Run the complete pipeline"""
    print("\n" + "="*80)
    print("STARTING FULL RAG PIPELINE")
    print("="*80)
    
    config.print_config()
    
    # Step 1: Load dataset (NO SPLIT)
    corpus, queries, qrels = step_1_load_dataset()
    
    # Step 2: Preprocess and create BoW
    bow_data, vocab, embeddings, doc_ids = step_2_preprocess_data(corpus)
    
    # Step 3: Train LDA
    lda_model, topics, lda_centers, topic_cost_matrix = step_3_train_lda(
        bow_data, embeddings, vocab
    )
    
    # Step 4: Generate BERT embeddings
    bert_embeddings = step_4_generate_bert_embeddings(corpus, doc_ids)
    
    # Step 5: Generate LOTT embeddings
    lott_embeddings, lott_topics = step_5_generate_lott_embeddings(bow_data)
    
    # Save metadata
    save_pipeline_metadata(doc_ids)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated artifacts:")
    print(f"  - Documents: {len(doc_ids)}")
    print(f"  - Queries: {len(queries)}")
    print(f"  - BERT embeddings: {bert_embeddings.shape}")
    print(f"  - LOTT embeddings: {lott_embeddings.shape}")
    print("\nAll embeddings are ready for RAG evaluation!")
    
    return {
        'corpus': corpus,
        'queries': queries,
        'qrels': qrels,
        'doc_ids': doc_ids,
        'bert_embeddings': bert_embeddings,
        'lott_embeddings': lott_embeddings
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