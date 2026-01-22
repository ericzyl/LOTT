import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import re
import bert

def basic_text_cleanup(text):
    # Avoiding aggressive cleaning and just multiple spaces replaced by single space
    text = re.sub(r'\s+', ' ', text)
    # Stripping leading/trailing whitespace
    text = text.strip()
    return text


def load_raw_text_dataset(file_path):
    texts = []
    labels = []
    
    print(f"Loading dataset from {file_path}...")
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # First word/token is label, rest is document text
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            
            label, text = parts
            # Remove quotes from labels if present
            label = label.strip('"')
            
            # Minimal preprocessing- just cleaning up white spaces
            text = basic_text_cleanup(text)
            
            texts.append(text)
            labels.append(label)
    
    # Convert string labels to integer indices
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = np.array([label_to_idx[label] for label in labels])
    
    print(f"Loaded {len(texts)} documents with {len(unique_labels)} unique labels")
    
    return texts, labels_int


def load_dataset_with_splits(train_file, test_file):
    texts_train = []
    labels_train = []
    texts_test = []
    labels_test = []
    
    print(f"Loading training data from {train_file}...")
    with open(train_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            label, text = parts
            label = label.strip('"')
            text = basic_text_cleanup(text)
            texts_train.append(text)
            labels_train.append(label)
    
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            label, text = parts
            label = label.strip('"')
            text = basic_text_cleanup(text)
            texts_test.append(text)
            labels_test.append(label)
    
    # Create unified label mapping from both train and test
    all_labels = sorted(set(labels_train + labels_test))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    y_train = np.array([label_to_idx[label] for label in labels_train])
    y_test = np.array([label_to_idx[label] for label in labels_test])
    
    print(f"Train: {len(texts_train)} documents, Test: {len(texts_test)} documents")
    print(f"Total unique labels: {len(all_labels)}")
    
    return texts_train, texts_test, y_train, y_test


# BERT Model Configs
BERT_MODELS = {
    'SBERT': 'sentence-transformers/all-MiniLM-L6-v2',
    'SBERT-large': 'sentence-transformers/all-mpnet-base-v2',
    'DistilBERT': 'distilbert-base-uncased',
    'RoBERTa': 'roberta-base',
    'BERT': 'bert-base-uncased',
}


# Dataset Configs
DATA_DIR = './Raw_Datasets/'

# has_splits=True means train/test files exist separately
# has_splits=False means single file needs to be split
DATASET_CONFIGS = {
    'bbcsport': {
        'file': DATA_DIR + 'all_bbcsport_by_line.txt',
        'has_splits': False
    },
    'twitter': {
        'file': DATA_DIR + 'all_twitter_by_line.txt',
        'has_splits': False
    },
    'amazon': {
        'file': DATA_DIR + 'all_amazon_by_line.txt',
        'has_splits': False
    },
    'classic': {
        'file': DATA_DIR + 'all_classic.txt',
        'has_splits': False
    },
    # Link to download not working currently
    # 'reuters': {
    #     'train_file': DATA_DIR + 'r8-train-all-terms.txt',
    #     'test_file': DATA_DIR + 'r8-test-all-terms.txt',
    #     'has_splits': True
    # },
    # 'ohsumed': {
    #     'file': DATA_DIR + 'ohsumed_by_line.txt',
    #     'has_splits': False,
    #     'first_n_classes': 10  # Using only first 10 classes for ohsumed dataset according to original WMD paper
    # },
    'ohsumed': {
        'train_file': DATA_DIR + 'train_ohsumed_by_line.txt',
        'test_file': DATA_DIR + 'test_ohsumed_by_line.txt',
        'has_splits': True
    },
    '20news': {
        'train_file': DATA_DIR + '20ng-train-all-terms.txt',
        'test_file': DATA_DIR + '20ng-test-all-terms.txt',
        'has_splits': True
    }
}

if __name__ == "__main__":
    all_results = []
    
    # Random seed for train/test split (for datasets without predefined splits)
    split_seed = 42
    test_size = 0.2  # 20% test size
    
    print("="*80)
    print("BERT Models on Raw Text Datasets")
    print("="*80)
    print(f"Train/Test split seed: {split_seed}")
    print(f"Test size: {test_size}")
    print(f"Using minimal preprocessing (no stopword removal, keeping natural text)")
    print("="*80)
    
    # Processing each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        try:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")
            
            # Loading dataset
            if config['has_splits']:
                # Loading pre-split data
                X_train, X_test, y_train, y_test = load_dataset_with_splits(
                    config['train_file'],
                    config['test_file']
                )
            else:
                # Loading single file and splitting
                texts, labels = load_raw_text_dataset(config['file'])
                
                # # Handling special dataset preprocessing
                # if dataset_name == 'ohsumed' and 'first_n_classes' in config:
                #     # Keeping only first N classes
                #     n_classes = config['first_n_classes']
                #     mask = labels < n_classes
                #     texts = [t for i, t in enumerate(texts) if mask[i]]
                #     labels = labels[mask]
                #     print(f"Filtered to first {n_classes} classes: {len(texts)} documents")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, 
                    test_size=test_size,
                    random_state=split_seed,
                    stratify=labels
                )
            
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            print(f"Number of classes: {len(np.unique(y_train))}")
            
            # Testing each BERT model
            for method, model_name in BERT_MODELS.items():
                print(f"\n{'-'*80}")
                print(f"Model: {method}")
                print(f"{'-'*80}")
                
                t_s = time.time()
                
                # Creating embeddings
                embedder = bert.BERTDocumentEmbedder(
                    model_name=model_name,
                    aggregation='mean'
                )
                
                print("Encoding training set...")
                X_train_emb = embedder.encode_documents(X_train, batch_size=16)
                
                print("Encoding test set...")
                X_test_emb = embedder.encode_documents(X_test, batch_size=16)
                
                # Training KNN classifier
                print("Training KNN classifier...")
                knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
                knn_classifier.fit(X_train_emb, y_train)
                
                # Predicting
                y_pred = knn_classifier.predict(X_test_emb)
                
                # Computing metrics
                test_error = 1 - accuracy_score(y_test, y_pred)
                runtime = time.time() - t_s
                
                print(f"\n{method} Results:")
                print(f"  Test error: {test_error:.6f}")
                print(f"  Test accuracy: {1-test_error:.6f}")
                print(f"  Runtime: {runtime:.2f} seconds")
                
                # Store results
                all_results.append({
                    'Dataset': dataset_name,
                    'Model': method,
                    'Test_Error': test_error,
                    'Test_Accuracy': 1 - test_error,
                    'Runtime_Seconds': runtime,
                    'Train_Size': len(X_train),
                    'Test_Size': len(X_test),
                    'Num_Classes': len(np.unique(y_train))
                })
        
        except Exception as e:
            print(f"\nError processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Computing final results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Printing summary table
    for dataset_name in DATASET_CONFIGS.keys():
        dataset_data = results_df[results_df['Dataset'] == dataset_name]
        if len(dataset_data) > 0:
            print(f"\n{dataset_name}:")
            for _, row in dataset_data.iterrows():
                print(f"  {row['Model']:15s}: Error = {row['Test_Error']:.6f}, "
                      f"Accuracy = {row['Test_Accuracy']:.6f}, Time = {row['Runtime_Seconds']:.2f}s")
    
    output_file = 'bert_raw_text_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    # Printing best models per dataset
    print("\n" + "="*80)
    print("BEST PERFORMING MODEL PER DATASET")
    print("="*80)
    for dataset_name in DATASET_CONFIGS.keys():
        dataset_data = results_df[results_df['Dataset'] == dataset_name]
        if len(dataset_data) > 0:
            best_model = dataset_data.loc[dataset_data['Test_Error'].idxmin()]
            print(f"{dataset_name:15s}: {best_model['Model']:15s} "
                  f"(Error = {best_model['Test_Error']:.6f})")
    
    print("\nAll datasets processed successfully!")