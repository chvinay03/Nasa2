# NASA Mission Intelligence RAG System — Project Report

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system for querying NASA space mission documents, including Apollo 11, Apollo 13, and the Challenger disaster. The system allows users to ask natural language questions and receive accurate, sourced answers backed by real NASA transcripts and reports.

---

## Components Implemented

### 1. `batch_eval.py`
Implements a CLI batch evaluation pipeline that:
- Loads test questions from `evaluation_dataset.txt` (one per line)
- Connects to a specified ChromaDB collection
- Runs each question through the full RAG pipeline: retrieval → LLM generation → RAGAS evaluation
- Prints a per-question summary table and aggregate (mean/min/max/stdev) for each metric
- Handles per-question errors gracefully with try/except so a single failure does not abort the batch

Run command:
```bash
python batch_eval.py \
  --questions ./evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --openai-key YOUR_KEY
```

### 3. `llm_client.py`
Implemented `generate_response()` which integrates with OpenAI's Chat Completions API. The function builds a message chain consisting of a NASA-specialist system prompt, context priming from retrieved documents, conversation history, and the current user question. Uses `temperature=0.3` for factual accuracy and `max_tokens=1000`.

### 4. `rag_client.py`
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

### Batch Evaluation Output (`batch_eval.py`)

The script was executed against the full 6-question `evaluation_dataset.txt`:

```
python batch_eval.py \
  --questions ./evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --openai-key sk-...
```

```
Loading questions from: ./evaluation_dataset.txt
  6 question(s) loaded.

Connecting to ChromaDB at './chroma_db_openai', collection 'nasa_space_missions_text'...
  Connected. Collection has 16576 document(s).

[1/6] What was the primary mission objective of Apollo 11?
  Retrieved 3 chunk(s).
  Answer preview: The primary mission objective of Apollo 11, as defined in the NASA Headquarters document OMSF M-D MA 500-11, was to perf...
  Scores: {'faithfulness': 1.0, 'answer_relevancy': 1.0, 'bleu_score': 0.0, 'rouge_score': 0.0}

[2/6] What problems did Apollo 13 encounter during its mission?
  Retrieved 3 chunk(s).
  Answer preview: Based on the provided context from NASA documents, there is no specific mention of the problems encountered by Apollo 13...
  Scores: {'faithfulness': 1.0, 'answer_relevancy': 0.0, 'bleu_score': 0.0, 'rouge_score': 0.0}

[3/6] What caused the Challenger disaster?
  Retrieved 3 chunk(s).
  Answer preview: Based on the provided context from NASA documents, the Challenger disaster (STS-51L) was caused by the failure of an O-r...
  Scores: {'faithfulness': 0.0, 'answer_relevancy': 0.873, 'bleu_score': 0.0, 'rouge_score': 0.0}

[4/6] Who were the crew members of Apollo 11?
  Retrieved 3 chunk(s).
  Answer preview: The crew members of Apollo 11 were Neil A. Armstrong (Commander), Michael Collins (Command Module Pilot), and Edwin E. A...
  Scores: {'faithfulness': 0.75, 'answer_relevancy': 1.0, 'bleu_score': 0.0, 'rouge_score': 0.0}

[5/6] How did the Apollo 13 crew survive after the oxygen tank explosion?
  Retrieved 3 chunk(s).
  Answer preview: The Apollo 13 crew survived after the oxygen tank explosion by isolating the damaged tank to conserve remaining oxygen f...
  Scores: {'faithfulness': 0.5, 'answer_relevancy': 1.0, 'bleu_score': 0.0, 'rouge_score': 0.0}

[6/6] What was the sequence of events during the Apollo 13 emergency?
  Retrieved 3 chunk(s).
  Answer preview: Based on the provided context from NASA documents, the sequence of events during the Apollo 13 emergency included: 1. T...
  Scores: {'faithfulness': 0.8, 'answer_relevancy': 0.9238, 'bleu_score': 0.0, 'rouge_score': 0.0}

=================================================================================================================================
BATCH EVALUATION RESULTS
=================================================================================================================================
Question                                                 Chunks    faithfulness  answer_relevancy      bleu_score     rouge_score
---------------------------------------------------------------------------------------------------------------------------------
What was the primary mission objective of Apollo 11?          3          1.0000          1.0000          0.0000          0.0000
What problems did Apollo 13 encounter during its mis...       3          1.0000          0.0000          0.0000          0.0000
What caused the Challenger disaster?                          3          0.0000          0.8730          0.0000          0.0000
Who were the crew members of Apollo 11?                       3          0.7500          1.0000          0.0000          0.0000
How did the Apollo 13 crew survive after the oxygen ...       3          0.5000          1.0000          0.0000          0.0000
What was the sequence of events during the Apollo 13...       3          0.8000          0.9238          0.0000          0.0000
---------------------------------------------------------------------------------------------------------------------------------
MEAN (all questions)                                                     0.6750          0.7995          0.0000          0.0000
=================================================================================================================================

Metric distributions:
  faithfulness: mean=0.6750  min=0.0000  max=1.0000  stdev=0.3791
  answer_relevancy: mean=0.7995  min=0.0000  max=1.0000  stdev=0.3951
  bleu_score: mean=0.0000  min=0.0000  max=0.0000  stdev=0.0000
  rouge_score: mean=0.0000  min=0.0000  max=0.0000  stdev=0.0000
```

**Notes on BLEU/ROUGE scores:** These metrics require a reference answer for comparison. Since `evaluation_dataset.txt` contains only questions (no gold-standard answers), BLEU and ROUGE report 0.0 — this is expected behaviour, not an error. Faithfulness and Answer Relevancy are the primary quality indicators for a RAG system without reference answers.

### Single-question evaluation scores (Apollo 13 question)
  - Faithfulness: 0.750
  - Answer Relevancy: 0.865
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
