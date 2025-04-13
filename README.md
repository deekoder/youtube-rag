# Building a RAG application from scratch

This is a step-by-step guide to building a simple RAG (Retrieval-Augmented Generation) application using Pinecone and OpenAI's API with langchain. The application will allow you to ask questions about any YouTube video. I have modified the original code quite a bit to adapt it to all the changes in langchain APIs and Pinecone APIs. Finally, it works.

## Setup

1. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/Scripts/activate
$ pip install -r requirements.txt
```

2. Create a free Pinecone account and get your API key from [here](https://www.pinecone.io/).

3. Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
```

## View the Python Notebook
Github sometimes does not display the notebook preview. Use this link instead https://nbviewer.org/github/deekoder/youtube-rag/blob/main/rag.ipynb



# YouTube RAG Application

This repository contains a Retrieval-Augmented Generation (RAG) application that uses YouTube video transcripts as a knowledge base to answer user queries.

> **Note:** This project was forked from an original repository with non-working code. The code has been fixed and updated to work with the latest versions of Pinecone and LangChain.

## Overview

This application demonstrates how to build a RAG system that:
1. Processes YouTube video transcripts
2. Embeds them using OpenAI's embedding model
3. Stores the embeddings in a Pinecone vector database
4. Retrieves relevant context based on user queries
5. Generates accurate responses using a language model

## Features

- YouTube transcript extraction
- Vector embedding with dimension handling
- Semantic search through Pinecone 
- Context-aware responses
- LangChain integration for streamlined RAG pipeline

## Prerequisites

- Python 3.9+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/deekoder/youtube-rag.git
cd youtube-rag
```

2. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/Scripts/activate
$ pip install -r requirements.txt
```

3. Create a free Pinecone account and get your API key from [here](https://www.pinecone.io/).

4. Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
```

## View the Python Notebook
Github sometimes does not display the notebook preview. Use this link instead https://nbviewer.org/github/deekoder/youtube-rag/blob/main/rag.ipynb

## Explanations

### 1. Process YouTube Videos

```python
from youtube_processor import extract_transcript

# Extract transcript from a YouTube video
video_id = "your_youtube_video_id"
transcript = extract_transcript(video_id)
```

### 2. Create Vector Database

```python
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index with correct dimensions
pc.create_index(
    name="youtube-rag-index-1536d",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### 3. Add Documents to Vector Store

```python
# Prepare documents
from langchain_core.documents import Document
documents = [Document(page_content=text, metadata={"source": "youtube"}) for text in texts]

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Get embeddings
embedding_vectors = embeddings.embed_documents([doc.page_content for doc in documents])

# Add to Pinecone
records = []
for i, (doc, embedding_vector) in enumerate(zip(documents, embedding_vectors)):
    vector_id = f"doc_{i}"
    record = {
        "id": vector_id,
        "values": embedding_vector,
        "metadata": {
            "text": doc.page_content,
            **doc.metadata
        }
    }
    records.append(record)

# Batch upsert
index.upsert(vectors=records)
```

### 4. Query the RAG System

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Set up vector store for retrieval
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    "Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {question}"
)

# Create model and chain
model = ChatOpenAI()
chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()} 
    | prompt 
    | model 
)

# Query the system
response = chain.invoke("What is the main topic discussed in this video?")
print(response.content)
```

## Common Challenges & Solutions

### Pinecone API Changes

The Pinecone API has undergone significant changes. If you encounter errors like:

```
TypeError: Pinecone.create_index() missing 1 required positional argument: 'spec'
```

Make sure you're using the latest syntax:

```python
pc.create_index(
    name="your-index-name",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### Dimension Mismatch

If you see an error like:

```
Vector dimension 1536 does not match the dimension of the index 1024
```

Ensure your index dimension matches your embedding model's output dimension (1536 for OpenAI's text-embedding-ada-002).

### LangChain Integration

When using LangChain with Pinecone, make sure the `text_key` parameter matches the field where your document content is stored:

```python
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # This must match what you used when storing documents
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
