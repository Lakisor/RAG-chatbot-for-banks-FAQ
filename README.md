# Banking FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for banking FAQs, featuring semantic search with FAISS and fine-tuned sentence transformers.

## Features

- Semantic search using FAISS and all-MiniLM-L6-v2
- Dual Encoder fine-tuning for improved relevance
- SQLite database for structured data storage
- GPT-2 for response generation
- Evaluation framework with precision@k metrics


## Project Structure

- `models.py`: Database models and connection handling
- `data_preparation.py`: Script to generate and embed FAQ data
- `retriever.py`: FAISS-based retriever with Dual Encoder
- `chatbot.py`: Main chatbot interface with GPT-2
- `evaluate.py`: Evaluation script for retrieval performance

## Performance

- Baseline Precision@3: ~77%
- 24% improvement in answer relevance after fine-tuning
