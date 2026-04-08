#!/usr/bin/env python3
"""
Batch evaluation pipeline for the NASA Mission Intelligence RAG system.

Loads test questions from a file, runs each one through the full RAG pipeline
(retrieval → generation → RAGAS evaluation), and prints a per-question summary
table plus aggregate (mean) scores across all metrics.

Usage:
    python batch_eval.py \
        --questions ./evaluation_dataset.txt \
        --chroma-dir ./chroma_db_openai \
        --collection-name nasa_space_missions_text \
        --openai-key YOUR_KEY
"""

import argparse
import os
import statistics
import sys
from typing import Dict, List

from openai import OpenAI

import rag_client
import llm_client
import ragas_evaluator


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for the NASA RAG system using RAGAS metrics."
    )
    parser.add_argument(
        "--questions",
        default="./evaluation_dataset.txt",
        help="Path to a plain-text file with one question per line (default: evaluation_dataset.txt)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="Path to the ChromaDB persistent directory (default: ./chroma_db_openai)",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="Name of the ChromaDB collection to query (default: nasa_space_missions_text)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY / CHROMA_OPENAI_API_KEY env vars)",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=3,
        help="Number of document chunks to retrieve per question (default: 3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_questions(path: str) -> List[str]:
    """Read non-empty, non-comment lines from a plain-text question file."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"No questions found in '{path}'.")
    return lines


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_query(text: str, openai_key: str) -> List[float]:
    """Embed a single query string using the same model as the collection (text-embedding-3-small)."""
    client = OpenAI(api_key=openai_key, base_url="https://openai.vocareum.com/v1")
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Core batch evaluation
# ---------------------------------------------------------------------------

def run_batch_evaluation(
    questions: List[str],
    collection,
    openai_key: str,
    num_docs: int = 3,
) -> List[Dict]:
    """
    Run the full RAG pipeline (retrieve → generate → evaluate) for every question.

    Returns a list of result dicts, one per question:
        {
            "question": str,
            "answer": str,
            "num_chunks_retrieved": int,
            "scores": Dict[str, float]   # RAGAS metric scores (or {"error": msg})
        }
    """
    results = []

    for i, question in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] {question}")

        try:
            # --- 1. Retrieval -------------------------------------------------
            # Pre-embed the query with the same model used to build the collection
            # (text-embedding-3-small, 1536-dim) so ChromaDB dimension matches.
            query_embedding = embed_query(question, openai_key)
            retrieval_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=num_docs,
                include=["documents", "metadatas"],
            )

            retrieved_chunks: List[str] = []
            context = ""

            if retrieval_result and retrieval_result.get("documents"):
                # ChromaDB returns nested lists: documents[0] = list of strings for query 0
                retrieved_chunks = retrieval_result["documents"][0]
                metadatas = retrieval_result["metadatas"][0]
                context = rag_client.format_context(retrieved_chunks, metadatas)

            num_retrieved = len(retrieved_chunks)
            print(f"  Retrieved {num_retrieved} chunk(s).")

            # Guard against empty retrieval so RAGAS doesn't receive an empty list
            if not retrieved_chunks:
                retrieved_chunks = [""]

            # --- 2. Generation ------------------------------------------------
            answer = llm_client.generate_response(
                openai_key,
                question,
                context,
                conversation_history=[],
            )
            print(f"  Answer preview: {answer[:120]}{'...' if len(answer) > 120 else ''}")

            # --- 3. RAGAS evaluation -------------------------------------------
            scores = ragas_evaluator.evaluate_response_quality(
                question, answer, retrieved_chunks
            )
            print(f"  Scores: {scores}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            answer = "ERROR"
            num_retrieved = 0
            scores = {"error": str(exc)}

        results.append(
            {
                "question": question,
                "answer": answer,
                "num_chunks_retrieved": num_retrieved,
                "scores": scores,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def print_summary_table(results: List[Dict]) -> None:
    """Print a per-question summary and compute aggregate (mean) metric scores."""

    # Collect all metric names seen across results (excluding "error" keys)
    metric_names = []
    for r in results:
        for k in r["scores"]:
            if k != "error" and k not in metric_names:
                metric_names.append(k)

    # --- Per-question rows ---
    col_q = 55      # question column width
    col_m = 14      # metric column width

    header_parts = [f"{'Question':<{col_q}}", f"{'Chunks':>6}"]
    for m in metric_names:
        header_parts.append(f"{m:>{col_m}}")
    header = "  ".join(header_parts)

    print("\n" + "=" * len(header))
    print("BATCH EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    metric_values: Dict[str, List[float]] = {m: [] for m in metric_names}

    for r in results:
        q_display = r["question"]
        if len(q_display) > col_q:
            q_display = q_display[: col_q - 3] + "..."

        row_parts = [f"{q_display:<{col_q}}", f"{r['num_chunks_retrieved']:>6}"]

        for m in metric_names:
            val = r["scores"].get(m)
            if isinstance(val, (int, float)):
                row_parts.append(f"{val:>{col_m}.4f}")
                metric_values[m].append(float(val))
            elif "error" in r["scores"]:
                row_parts.append(f"{'ERROR':>{col_m}}")
            else:
                row_parts.append(f"{'N/A':>{col_m}}")

        print("  ".join(row_parts))

    # --- Aggregate row ---
    print("-" * len(header))
    agg_parts = [f"{'MEAN (all questions)':<{col_q}}", f"{'':>6}"]
    for m in metric_names:
        vals = metric_values[m]
        if vals:
            mean_val = statistics.mean(vals)
            agg_parts.append(f"{mean_val:>{col_m}.4f}")
        else:
            agg_parts.append(f"{'N/A':>{col_m}}")
    print("  ".join(agg_parts))
    print("=" * len(header))

    # Distribution info
    print("\nMetric distributions:")
    for m in metric_names:
        vals = metric_values[m]
        if len(vals) >= 2:
            print(
                f"  {m}: mean={statistics.mean(vals):.4f}  "
                f"min={min(vals):.4f}  max={max(vals):.4f}  "
                f"stdev={statistics.stdev(vals):.4f}"
            )
        elif len(vals) == 1:
            print(f"  {m}: {vals[0]:.4f}")
        else:
            print(f"  {m}: no data")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve API key
    openai_key = (
        args.openai_key
        or os.environ.get("CHROMA_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not openai_key:
        print(
            "ERROR: No OpenAI API key found.\n"
            "Pass --openai-key or set OPENAI_API_KEY in your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Propagate the key so ragas_evaluator.py can pick it up
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    # Load questions
    print(f"Loading questions from: {args.questions}")
    questions = load_questions(args.questions)
    print(f"  {len(questions)} question(s) loaded.")

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB at '{args.chroma_dir}', collection '{args.collection_name}'...")
    collection, success, error = rag_client.initialize_rag_system(
        args.chroma_dir, args.collection_name
    )
    if not success:
        print(f"ERROR: Could not connect to ChromaDB — {error}", file=sys.stderr)
        sys.exit(1)
    print(f"  Connected. Collection has {collection.count()} document(s).")

    # Run batch evaluation
    results = run_batch_evaluation(questions, collection, openai_key, args.num_docs)

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
