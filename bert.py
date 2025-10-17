import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

class BERTDocumentEmbedder: # Class to generate Document Embeddings using different BERT Models
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 aggregation='mean', device=None):
        self.model_name = model_name
        self.aggregation = aggregation
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def mean_pooling(self, token_embeddings, attention_mask): # Attention Mask used for correct Average Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Padding Tokens set to Large Negative Value
        return torch.max(token_embeddings, 1)[0]
    
    def cls_pooling(self, token_embeddings): # Using CLS Token Embedding
        return token_embeddings[:, 0]
    
    def encode_documents(self, documents, batch_size=16, max_length=512): # Encoding List of Documents into Embeddings
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
                batch_docs = documents[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                
                # Applying Pooling Strategy
                if self.aggregation == 'mean':
                    batch_embeddings = self.mean_pooling(token_embeddings, encoded['attention_mask'])
                elif self.aggregation == 'max':
                    batch_embeddings = self.max_pooling(token_embeddings, encoded['attention_mask'])
                elif self.aggregation == 'cls':
                    batch_embeddings = self.cls_pooling(token_embeddings)
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation}")
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


def bow_to_text(bow_data, vocab): # Converting Bag-of-Words to Text
    documents = []
    for doc_bow in bow_data:
        words = []
        for idx, count in enumerate(doc_bow):
            if count > 0:
                words.extend([vocab[idx]] * int(count))
        documents.append(' '.join(words))
    return documents


def create_bert_embeddings(bow_data, vocab, model_name='sentence-transformers/all-MiniLM-L6-v2',
                          aggregation='mean', batch_size=16):
    # Converting BOW to text
    documents = bow_to_text(bow_data, vocab)
    
    # Initializing Embedder and encoding
    embedder = BERTDocumentEmbedder(model_name=model_name, aggregation=aggregation)
    embeddings = embedder.encode_documents(documents, batch_size=batch_size)
    
    return embeddings

# Model Configs
BERT_MODELS = {
    'SBERT': 'sentence-transformers/all-MiniLM-L6-v2',
    'SBERT-large': 'sentence-transformers/all-mpnet-base-v2',
    'DistilBERT': 'distilbert-base-uncased',
    'RoBERTa': 'roberta-base',
    'BERT': 'bert-base-uncased',
}