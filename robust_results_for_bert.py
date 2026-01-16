import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import pandas as pd

from data import loader
import bert

# data_path = '/kaggle/input/bert-lott-exps-datasets/'
# embeddings_path = '/kaggle/input/bert-lott-exps-datasets/glove.6B.300d.txt'
data_path = './data/'
embeddings_path = './data/dolma_300_2024_1.2M.100_combined.txt'

datasets = [
    'bbcsport-emd_tr_te_split.mat',
    'twitter-emd_tr_te_split.mat',
    'r8-emd_tr_te3.mat',
    'amazon-emd_tr_te_split.mat',
    'classic-emd_tr_te_split.mat',
    'ohsumed-emd_tr_te_ix.mat',
    '20ng2_500-emd_tr_te.mat',
    'recipe2-emd_tr_te_split.mat'
]

bert_models = ['SBERT', 'SBERT-large', 'DistilBERT', 'RoBERTa', 'BERT']

p = 1
split_seed = 0  # Fixed Seed for train/test split
num_permutations = 100  # Number of Word order Permutations

all_results = []
detailed_results = []

def bow_to_text_with_seed(bow_data, vocab, seed):
    np.random.seed(seed)
    documents = []
    
    for doc_bow in bow_data:
        words = []
        for idx, count in enumerate(doc_bow):
            if count > 0:
                words.extend([vocab[idx]] * int(count))
        
        # Permuting Word Order based on Seed
        np.random.shuffle(words)
        documents.append(' '.join(words))
    
    return documents


def create_bert_embeddings_with_seed(bow_data, vocab, model_name, aggregation, batch_size, seed):
    # Converting BoW to Text with Permuted Word Order
    documents = bow_to_text_with_seed(bow_data, vocab, seed)
    
    embedder = bert.BERTDocumentEmbedder(model_name=model_name, aggregation=aggregation)
    embeddings = embedder.encode_documents(documents, batch_size=batch_size)
    
    return embeddings


print("<>"*40)
print("BERT Models Performance with Word Order Permutations")
print("<>"*40)
print(f"Number of permutations per model: {num_permutations}")
print(f"Train/Test split seed: {split_seed}")
print("<>"*40)

for data_name in datasets:
    try:
        print(f"\n{'<>'*40}")
        print(f"Dataset: {data_name}")
        print(f"{'<>'*40}")

        data = loader(data_path + data_name, embeddings_path, p=p)
        bow_data, y = data['X'], data['y']
        vocab = data['vocab']
        topic_proportions = data['proportions']

        # Splitting Data with fixed Seed for consistency
        bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(
            bow_data, topic_proportions, y, random_state=split_seed
        )

        dataset_results = {model: [] for model in bert_models}

        for method in bert_models:
            print(f"\n{'-'*80}")
            print(f"Model: {method}")
            print(f"{'-'*80}")
            
            model_name = bert.BERT_MODELS[method]
            
            # Testing with different Word Order Permutations
            for perm_seed in range(num_permutations):
                t_s = time.time()
                
                # Creating Embeddings with Permuted Word Order
                X_train_bert = create_bert_embeddings_with_seed(
                    bow_train, vocab,
                    model_name=model_name,
                    aggregation='mean',
                    batch_size=128,
                    seed=perm_seed
                )
                X_test_bert = create_bert_embeddings_with_seed(
                    bow_test, vocab,
                    model_name=model_name,
                    aggregation='mean',
                    batch_size=128,
                    seed=perm_seed
                )
                
                knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
                knn_classifier.fit(X_train_bert, y_train)
                y_pred = knn_classifier.predict(X_test_bert)
                
                test_error = 1 - accuracy_score(y_test, y_pred)
                runtime = time.time() - t_s
                
                dataset_results[method].append(test_error)
                
                detailed_results.append({
                    'Dataset': data_name,
                    'Model': method,
                    'Permutation_Seed': perm_seed,
                    'Test_Error': test_error,
                    'Runtime_Seconds': runtime
                })
                
                if (perm_seed + 1) % 1 == 0:
                    mean_error = np.mean(dataset_results[method])
                    std_error = np.std(dataset_results[method])
                    print(f"  Permutation {perm_seed + 1}/{num_permutations}: "
                          f"Current error = {test_error:.6f}, "
                          f"Mean = {mean_error:.6f}, Std = {std_error:.6f}, "
                          f"Time = {runtime:.2f}s")
            
            # Computing Statistics for a particular Model on a particular Dataset
            mean_error = np.mean(dataset_results[method])
            std_error = np.std(dataset_results[method])
            min_error = np.min(dataset_results[method])
            max_error = np.max(dataset_results[method])
            
            print(f"\n{method} Summary Statistics:")
            print(f"  Mean test error: {mean_error:.6f}")
            print(f"  Std deviation:   {std_error:.6f}")
            print(f"  Min test error:  {min_error:.6f}")
            print(f"  Max test error:  {max_error:.6f}")
            
            all_results.append({
                'Dataset': data_name,
                'Model': method,
                'Mean_Error': mean_error,
                'Std_Error': std_error,
                'Min_Error': min_error,
                'Max_Error': max_error,
                'Num_Permutations': num_permutations
            })

    except Exception as e:
        print(f"\nError processing dataset {data_name}: {e}")
        import traceback
        traceback.print_exc()

# Summary Dataframe
print("\n" + "<>"*40)
print("FINAL SUMMARY - MEAN TEST ERRORS ACROSS ALL PERMUTATIONS")
print("="*80)

results_df = pd.DataFrame(all_results)

for data_name in datasets:
    dataset_data = results_df[results_df['Dataset'] == data_name]
    if len(dataset_data) > 0:
        print(f"\n{data_name}:")
        for _, row in dataset_data.iterrows():
            print(f"  {row['Model']:15s}: {row['Mean_Error']:.6f} ± {row['Std_Error']:.6f} "
                  f"(min={row['Min_Error']:.6f}, max={row['Max_Error']:.6f})")

output_file = 'bert_permutation_results.csv'
results_df.to_csv(output_file, index=False)

detailed_output_file = 'bert_permutation_detailed_results.csv'
detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(detailed_output_file, index=False)

print(f"\n{'<>'*40}")
print(f"Summary results saved to: {output_file}")
print(f"Detailed results (with seeds) saved to: {detailed_output_file}")
print("<>"*40)

print("\n" + "<>"*40)
print("BEST PERFORMING MODEL PER DATASET (by mean error)")
print("="*80)
for data_name in datasets:
    dataset_data = results_df[results_df['Dataset'] == data_name]
    if len(dataset_data) > 0:
        best_model = dataset_data.loc[dataset_data['Mean_Error'].idxmin()]
        print(f"{data_name:40s}: {best_model['Model']:15s} "
              f"({best_model['Mean_Error']:.6f} ± {best_model['Std_Error']:.6f})")

print("\nAll datasets processed successfully!")