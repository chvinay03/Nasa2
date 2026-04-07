# NASA Mission Intelligence RAG System — Project Report

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system for querying NASA space mission documents, including Apollo 11, Apollo 13, and the Challenger disaster. The system allows users to ask natural language questions and receive accurate, sourced answers backed by real NASA transcripts and reports.

---

## Components Implemented

### 1. `llm_client.py`
Implemented `generate_response()` which integrates with OpenAI's Chat Completions API. The function builds a message chain consisting of a NASA-specialist system prompt, context priming from retrieved documents, conversation history, and the current user question. Uses `temperature=0.3` for factual accuracy and `max_tokens=1000`.

### 2. `rag_client.py`
Implemented four functions:
- `discover_chroma_backends()` — scans the project directory for ChromaDB collections
- `initialize_rag_system()` — connects to a specified collection
- `retrieve_documents()` — performs semantic search with optional mission filtering (apollo_11, apollo_13, challenger)
- `format_context()` — formats retrieved chunks into structured context with source attribution

### 3. `embedding_pipeline.py`
Implemented the `ChromaEmbeddingPipelineTextOnly` class with:
- Overlapping text chunking with sentence boundary detection
- OpenAI `text-embedding-3-small` model for embeddings
- ChromaDB persistent storage with cosine similarity
- Skip/update/replace modes for document management
- Metadata extraction (mission, source, document category)
- CLI interface for running and managing the pipeline

### 4. `ragas_evaluator.py`
Implemented `evaluate_response_quality()` using four RAGAS metrics:
- **Faithfulness** — measures if the answer is grounded in the retrieved context
- **Answer Relevancy** — measures how well the answer addresses the question
- **BLEU Score** — word overlap metric
- **ROUGE Score** — recall-based overlap metric

---

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Vocareum API key (`voc-`) not valid on local machine | Obtained a real `sk-` OpenAI key from platform.openai.com |
| Virtual environment not activating | Used `.venv/bin/activate` instead of `venv/bin/activate` |
| Missing `sacrebleu` and `rouge_score` packages | Installed with `pip install sacrebleu rouge_score` |
| RAGAS async compatibility with Streamlit | Used `nest_asyncio.apply()` to handle running event loops |
| Large NASA files creating thousands of chunks | Used chunk_size=500 with chunk_overlap=100 for balanced retrieval |

---

## Results

- **Total documents processed:** 12 files (6 Apollo 11, 3 Apollo 13, 3 Challenger)
- **Total chunks embedded:** 16,576 documents in ChromaDB
- **Sample evaluation scores (Apollo 13 question):**
  - Faithfulness: 0.429
  - Answer Relevancy: 0.780
  - BLEU Score: 0.000 (no reference answer provided)
  - ROUGE Score: 0.000 (no reference answer provided)

---

## Sample Queries and Responses

**Q: What problems did Apollo 13 encounter during its mission?**

> During the Apollo 13 mission, the spacecraft encountered an oxygen tank explosion that caused a loss of electrical power, loss of cabin heat, shortage of drinkable water, and limited use of the propulsion system. This led to the mission being aborted and the crew having to use the Lunar Module as a "lifeboat" to return safely to Earth. *(Source: NASA - Apollo 13 Mission Report)*

---

## Additional Features

- Mission-specific filtering in the sidebar (filter by Apollo 11, Apollo 13, or Challenger)
- Real-time RAGAS evaluation scores displayed after every response
- Conversation history maintained across turns
- Configurable retrieval count (1–10 documents)
- Support for GPT-3.5-turbo and GPT-4 model selection
