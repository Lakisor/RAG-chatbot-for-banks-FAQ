import numpy as np
from models import Database, FAQ
from sentence_transformers import SentenceTransformer
import random
from faker import Faker

class FAQGenerator:
    def __init__(self):
        self.fake = Faker()
        self.questions = self._generate_base_questions()
        self._add_variations()
    
    def _generate_base_questions(self):
        return [
            # Account Management
            ("How do I open a savings account?", 
             "You can open a savings account online or by visiting any of our branches with your ID and proof of address. Minimum opening deposit is $25."),
            ("What's the minimum balance for a checking account?", 
             "Our basic checking account requires a minimum balance of $100 to avoid the $10 monthly maintenance fee."),
            ("How do I close my bank account?",
             "You can close your account by visiting a branch with valid ID and signing an account closure form. Ensure your balance is zero before closing."),
            
            # Cards and Payments
            ("How do I report a lost or stolen card?", 
             "Call our 24/7 customer service at 1-800-BANK-123 immediately to report a lost or stolen card. We'll block the card and issue a replacement."),
            ("How do I activate my new debit card?",
             "Activate your card by calling 1-800-555-0199 or through our mobile app. You'll need the card number and your account information."),
            ("What are the foreign transaction fees?",
             "We charge a 3% foreign transaction fee on all purchases made outside the country. There's also a 1% currency conversion fee."),
            
            # Loans and Credit
            ("How do I apply for a personal loan?",
             "You can apply online, through our mobile app, or at any branch. You'll need proof of income, employment verification, and a valid ID."),
            ("What's the current mortgage interest rate?",
             f"Our current 30-year fixed mortgage rate is {random.uniform(5.5, 6.5):.2f}% APR. Rates vary based on credit score and down payment."),
            ("What's the minimum credit score for a car loan?",
             "The minimum credit score for our auto loans is 620. Better rates are available for scores above 720."),
            
            # Digital Banking
            ("How do I reset my online banking password?",
             "Click 'Forgot Password' on the login page and follow the instructions. You'll need access to your registered email or phone number."),
            ("Is mobile check deposit available?",
             "Yes, you can deposit checks using our mobile app. Endorse the check and take clear photos of the front and back."),
            ("How do I set up account alerts?",
             "Log in to online banking, go to 'Alerts' and choose the notifications you'd like to receive via email or text message."),
            
            # Transactions and Limits
            ("What's the daily ATM withdrawal limit?",
             "The standard daily ATM withdrawal limit is $500. You can request an increase up to $1,000 through online banking or by calling customer service."),
            ("How long do pending transactions take to clear?",
             "Most pending transactions clear within 1-3 business days. Debit card transactions typically clear faster than checks."),
            ("How do I dispute a transaction?",
             "Call customer service immediately if you notice unauthorized transactions. You can also dispute transactions through online banking under 'Dispute Center'.")
        ]
    
    def _add_variations(self):
        variations = []
        for q, a in self.questions:
            templates = [
                (f"{q}", a),
                (f"{q.lower()}", a),
                (f"{q} {random.choice(['please', 'thanks', 'thank you', ''])}", a),
                (f"{random.choice(['Can you tell me', 'Do you know', 'Could you explain'])} {q.lower()}", a),
                (f"{random.choice(['I need to know', 'I want to find out'])} {q.lower()}", a),
                (f"{random.choice(['Details about', 'Information about', 'Help with'])} {q.split('?')[0].lower()}", a),
                (f"{q.replace('?', '')} - {random.choice(['need help', 'urgent', 'quick question'])}", a)
            ]
            variations.extend(templates)
        
        for q, a in self.questions:
            if random.random() > 0.7:
                q = q.replace("you're", "your").replace("you are", "u r").replace("for", "4")
                variations.append((q, a))
        
        self.questions = list(set(self.questions + variations))

class DataProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.db = Database()
    
    def generate_embeddings(self):
        generator = FAQGenerator()
        session = self.db.get_session()
        
        try:
            session.query(FAQ).delete()
            session.commit()
            
            batch_size = 32
            questions = generator.questions
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                batch_questions = [q for q, _ in batch]
                embeddings = self.model.encode(batch_questions, show_progress_bar=False, convert_to_numpy=True)
                
                for (q, a), emb in zip(batch, embeddings):
                    faq = FAQ(question=q, answer=a)
                    faq.set_embedding(emb)
                    session.add(faq)
                
                session.commit()
                
        except Exception as e:
            session.rollback()
            print(f"Error: {str(e)}")
            raise e
        finally:
            session.close()

if __name__ == "__main__":
    try:
        import faker
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "faker"])
    
    processor = DataProcessor()
    processor.generate_embeddings()
