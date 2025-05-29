from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from retriever import DualEncoderRetriever
import numpy as np

class RAGChatbot:
    def __init__(self, model_name='gpt2'):
        self.retriever = DualEncoderRetriever()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, query, max_length=150, temperature=0.7, top_k=50, top_p=0.9):
        relevant_faqs = self.retriever.get_relevant_faqs(query, k=2)
        
        if not relevant_faqs:
            return "I'm sorry, I couldn't find any relevant information to answer your question."
        
        context = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in relevant_faqs])
        
        prompt = f"""Based on the following banking FAQs, answer the user's question.
        
        {context}
        
        Question: {query}
        Answer:"""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

def interactive_chat():
    chatbot = RAGChatbot()
    print("Banking FAQ Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
            
        response = chatbot.generate_response(query)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    interactive_chat()
