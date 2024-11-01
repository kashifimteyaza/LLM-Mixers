
A repository showcasing practical applications of Large Language Models (LLMs) focusing on recommendation systems, app review analysis, and RAG.
📱 Projects
# 1. App Review Scraper
A Python-based tool for scraping and analyzing app reviews from Google Play Store. Built using Google Colab for easy access and execution.
🌟 Overview
This tool helps researchers and analysts collect digital social traces in the form of app reviews. It's particularly useful for:

Market Research
User Experience Analysis
Customer Feedback Analysis
App Performance Monitoring
Sentiment Analysis

# 🛠️ Key Features

Scrapes reviews from Google Play Store
Collects review content, ratings, and timestamps
Supports sorting by newest/oldest reviews
Exports data to CSV and Excel formats
Includes user metadata (reviewId, userName, etc.)
Configurable review count and language settings

# 📝 Notes
Large numbers of reviews may take longer to fetch
Some reviews might not have all fields populated
Export files will be saved in the current working directory
When using in Google Colab, files can be downloaded from the file explorer. 

# 2. LLM-based Recommendation System (LLM as RecSys (1).py)
A content-based movie recommendation system using LLMs and FAISS for efficient similarity search.
Features

Generates dense embeddings for movie descriptions using Llama 2
Uses FAISS for efficient nearest neighbor search
Includes caching mechanism for embeddings
Parallel processing for batch embedding generation
Content-based recommendations based on movie metadata

# Technical Details

Input Data: Netflix titles dataset with fields:

type
title
director
cast
release_year
listed_in (genres)
description


Dependencies:
Copynumpy
pandas
faiss-cpu
requests
tqdm
pickle

# Key Components:

Text Representation Creation
Embedding Generation (using Llama 2)
FAISS Index Creation
Similarity Search


# Usage
pythonCopy# Setup Ollama with Llama 2 model locally
# Ensure Ollama is running on localhost:11434

# Run the script
python "LLM as RecSys (1).py"

# The code will:
# 1. Load Netflix dataset
# 2. Generate embeddings (or load from cache)
# 3. Create FAISS index
# 4. Allow searching for similar movies
Performance Optimization

Implements batch processing
Uses ThreadPoolExecutor for parallel embedding generation
Caches embeddings to avoid regeneration
Uses FAISS for efficient similarity search

# Notes 📝

The first run will take longer due to embedding generation.
Subsequent runs will use cached embeddings.
Adjust batch_size and num_workers based on your system's capabilities.
Keep Ollama running while using the recommendation system.

# 3 📄 PDF Chatbot with LangChain
An interactive chatbot that can understand and answer questions about PDF documents using LangChain and OpenAI. Built with RAG (Retrieval-Augmented Generation) for accurate, context-aware responses.
🌟 Features

Upload and process PDF documents
Interactive question-answering interface
Multiple chain types for different querying strategies
Configurable number of relevant chunks for context
Real-time response generation
Source reference for answers
Clean, modern UI with Panel

# 🛠️ Prerequisites

Python 3.7+
OpenAI API key
Google Colab or local Jupyter environment

# UI Components

File upload interface
API key input
Question text editor
Chain type selector
Chunk number slider
Response display with source references

# ⚙️ Configuration Options

# Chain Types:

stuff: Best for small documents
map_reduce: Efficient for large documents
refine: Detailed, iterative answers
map_rerank: Prioritizes most relevant context


# Chunk Settings:
Adjust number of relevant chunks (1-5)
Higher numbers provide more context but may be slower
