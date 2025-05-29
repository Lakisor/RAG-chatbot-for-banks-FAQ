import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from models import Database, FAQ
import torch
import time
from typing import List, Dict, Any, Optional

class DualEncoderRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        """
        Initialize the retriever with FAISS index and sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
            use_gpu: Whether to use GPU for FAISS if available
        """
        self.db = Database()
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.faq_data = []  # Store (id, question, answer) tuples
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._build_index()
    
    def _build_index(self) -> None:
        """Build or rebuild the FAISS index from database"""
        start_time = time.time()
        session = self.db.get_session()
        try:
            faqs = session.query(FAQ).all()
            if not faqs:
                raise ValueError("No FAQs found in database. Please run data_preparation.py first.")
            
            embeddings = []
            self.faq_data = []
            
            for faq in faqs:
                embedding = faq.get_embedding()
                if embedding is not None:
                    embeddings.append(embedding)
                    self.faq_data.append({
                        'id': faq.id,
                        'question': faq.question,
                        'answer': faq.answer
                    })
            
            if not embeddings:
                raise ValueError("No valid embeddings found in the database.")
            
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.ef_construction = 40
            self.index.hnsw.ef_search = 16
            
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            self.index.add(embeddings)
            
        except Exception as e:
            raise e
        finally:
            session.close()
    
    def get_relevant_faqs(self, query: str, k: int = 3, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve relevant FAQs for a given query
        
        Args:
            query: The search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of dictionaries with 'question', 'answer', and 'score' keys
        """
        if self.index is None or not self.faq_data:
            self._build_index()
        
        try:
            query_embedding = self.model.encode([query], show_progress_bar=False)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            k = min(k, len(self.faq_data))
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx >= 0 and (1.0 - score) >= score_threshold:
                    faq = self.faq_data[idx]
                    results.append({
                        'id': faq['id'],
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'score': float(1.0 - score)
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
    
    def finetune_dual_encoder(self, queries: List[str], relevant_faq_indices: List[List[int]], 
                            num_epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5) -> None:
        """
        Fine-tune the dual encoder model using contrastive learning
        
        Args:
            queries: List of query strings
            relevant_faq_indices: List of lists, where each sublist contains relevant FAQ indices
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for the optimizer
        """
        from torch.utils.data import DataLoader, Dataset
        import torch.nn.functional as F
        
        class FAQDataset(Dataset):
            def __init__(self, queries, relevant_indices, faq_data):
                self.queries = queries
                self.relevant_indices = relevant_indices
                self.faq_data = faq_data
                
                self.pairs = []
                for query, rel_indices in zip(queries, relevant_indices):
                    for idx in rel_indices:
                        if idx < len(faq_data):
                            self.pairs.append((query, faq_data[idx]['question'], 1.0))
                    
                    neg_indices = np.random.choice(
                        [i for i in range(len(faq_data)) if i not in rel_indices],
                        min(len(rel_indices) * 2, len(faq_data) - len(rel_indices)),
                        replace=False
                    )
                    for idx in neg_indices:
                        self.pairs.append((query, faq_data[idx]['question'], 0.0))
            
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                return self.pairs[idx]
        
        dataset = FAQDataset(queries, relevant_faq_indices, self.faq_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.model.train()
        device = next(self.model.parameters()).device
        

        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                queries, docs, labels = batch
                labels = labels.float().to(device)
                
                query_embeddings = self.model.encode(queries, convert_to_tensor=True)
                doc_embeddings = self.model.encode(docs, convert_to_tensor=True)
                
                scores = F.cosine_similarity(query_embeddings, doc_embeddings)
                
                loss = criterion(scores, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        print("Fine-tuning complete. Rebuilding index...")
        self._build_index()


if __name__ == "__main__":
    retriever = DualEncoderRetriever()
    
    test_queries = [
        "How do I open a new account?",
        "What's the interest rate for savings?",
        "How to report a stolen card?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.get_relevant_faqs(query, k=3)
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res['score']:.3f}] {res['question']}")
            print(f"   {res['answer']}")
            print("-" * 80)
