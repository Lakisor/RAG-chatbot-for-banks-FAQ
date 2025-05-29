import numpy as np
from retriever import DualEncoderRetriever
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def evaluate_retrieval(self, test_queries, relevant_faq_indices, k=3):
        """
        Evaluate retrieval performance
        
        Args:
            test_queries: List of test query strings
            relevant_faq_indices: List of lists, where each sublist contains relevant FAQ indices for the corresponding query
            k: Number of results to consider for evaluation
            
        Returns:
            Dictionary containing precision@k and other metrics
        """
        precisions = []
        recalls = []
        
        for i, (query, true_relevant_indices) in enumerate(zip(test_queries, relevant_faq_indices)):
            results = self.retriever.get_relevant_faqs(query, k=k)
            retrieved_ids = [result['id'] for result in results]
            
            true_positives = len(set(retrieved_ids) & set(true_relevant_indices))
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            recall = true_positives / len(true_relevant_indices) if true_relevant_indices else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) 
                    if (avg_precision + avg_recall) > 0 else 0
        }

def create_test_data():
    """
    Create test data for evaluation using actual FAQ data from the database
    Returns tuple of (test_queries, relevant_faq_ids)
    """
    from models import Database, FAQ
    
    db = Database()
    session = db.get_session()
    faqs = session.query(FAQ).all()
    
    test_data = [
        ("How can I open a new account?", [faq.id for faq in faqs if 'open' in faq.question.lower() and 'account' in faq.question.lower()]),
        ("What's needed for a loan application?", [faq.id for faq in faqs if 'loan' in faq.question.lower() and 'apply' in faq.question.lower()]),
        ("How to reset my online banking password?", [faq.id for faq in faqs if 'reset' in faq.question.lower() and 'password' in faq.question.lower()]),
        ("What are your branch hours?", [faq.id for faq in faqs if 'hours' in faq.question.lower() or 'open' in faq.question.lower()]),
        ("How to report a stolen card?", [faq.id for faq in faqs if 'stolen' in faq.question.lower() or 'lost' in faq.question.lower()]),
        ("What's the minimum balance for checking?", [faq.id for faq in faqs if 'minimum' in faq.question.lower() and 'balance' in faq.question.lower()]),
        ("How to set up direct deposit?", [faq.id for faq in faqs if 'direct deposit' in faq.question.lower()]),
        ("What's the daily ATM limit?", [faq.id for faq in faqs if 'ATM' in faq.question or 'withdrawal' in faq.question]),
        ("How to order new checks?", [faq.id for faq in faqs if 'order' in faq.question.lower() and 'check' in faq.question.lower()]),
        ("Where can I find the routing number?", [faq.id for faq in faqs if 'routing' in faq.question.lower() or 'account number' in faq.question.lower()])
    ]
    
    test_data = [(q, ids) for q, ids in test_data if ids]
    
    if not test_data:
        raise ValueError("No matching FAQs found for test queries. Please check your database.")
    
    test_queries = [item[0] for item in test_data]
    relevant_faq_ids = [item[1] for item in test_data]
    
    session.close()
    return test_queries, relevant_faq_ids

def evaluate_retriever():
    """Evaluate the retriever's performance"""
    retriever = DualEncoderRetriever()
    test_queries, relevant_faq_indices = create_test_data()
    evaluator = Evaluator(retriever)
    return evaluator.evaluate_retrieval(test_queries, relevant_faq_indices, k=3)

if __name__ == "__main__":
    evaluate_retriever()
