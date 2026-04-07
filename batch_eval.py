#!/usr/bin/env python3
"""
Batch Evaluation Script for NASA Mission Intelligence RAG System
Runs a set of questions through the full RAG pipeline and evaluates
each response using RAGAS metrics, then prints a summary table.
"""

import os
import argparse
from pathlib import Path

import rag_client
import llm_client
import ragas_evaluator


def load_questions(filepath: str):
    """Load questions from a text file, one per line."""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            q = line.strip()
            if q:
                questions.append(q)
    return questions


def run_batch_evaluation(questions, collection, openai_key: str,
                         model: str = "gpt-3.5-turbo", num_docs: int = 3):
    """
    Run each question through retrieval + generation + evaluation.
    Returns a list of result dicts.
    """
    results = []

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] Evaluating: {question}")

        # Retrieve relevant document chunks
        retrieval_result = rag_client.retrieve_documents(collection, question, num_docs)

        retrieved_chunks = []
        context = ""
        if retrieval_result and retrieval_result.get("documents"):
            retrieved_chunks = retrieval_result["documents"][0]
            context = rag_client.format_context(
                retrieval_result["documents"][0],
                retrieval_result["metadatas"][0]
            )

        # Generate answer
        answer = llm_client.generate_response(
            openai_key, question, context, conversation_history=[], model=model
        )

        # Evaluate with RAGAS
        scores = ragas_evaluator.evaluate_response_quality(question, answer, retrieved_chunks)

        results.append({
            "question": question,
            "answer": answer,
            "num_chunks_retrieved": len(retrieved_chunks),
            "scores": scores
        })

        print(f"  Scores: {scores}")

    return results


def print_summary_table(results):
    """Print a formatted summary table of all evaluation results."""
    metric_names = ["faithfulness", "answer_relevancy", "bleu_score", "rouge_score"]

    print("\n" + "=" * 90)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 90)

    header = f"{'#':<4} {'Question':<50} " + " ".join(f"{m[:10]:<12}" for m in metric_names)
    print(header)
    print("-" * 90)

    totals = {m: [] for m in metric_names}

    for i, result in enumerate(results):
        scores = result["scores"]
        question_short = result["question"][:48] + ".." if len(result["question"]) > 48 else result["question"]
        row = f"{i + 1:<4} {question_short:<50} "
        for m in metric_names:
            val = scores.get(m, "N/A")
            if isinstance(val, float):
                row += f"{val:<12.4f}"
                totals[m].append(val)
            else:
                row += f"{'ERR':<12}"
        print(row)

    print("-" * 90)

    avg_row = f"{'AVG':<4} {'':50} "
    for m in metric_names:
        if totals[m]:
            avg_row += f"{sum(totals[m]) / len(totals[m]):<12.4f}"
        else:
            avg_row += f"{'N/A':<12}"
    print(avg_row)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of the NASA RAG system')
    parser.add_argument('--questions', default='./evaluation_dataset.txt',
                        help='Path to questions file (one question per line)')
    parser.add_argument('--openai-key', default=os.environ.get("OPENAI_API_KEY"),
                        help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai',
                        help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text',
                        help='ChromaDB collection name')
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        help='OpenAI model to use for generation')
    parser.add_argument('--num-docs', type=int, default=3,
                        help='Number of documents to retrieve per question')
    args = parser.parse_args()

    if not args.openai_key:
        print("ERROR: No OpenAI API key found. Set OPENAI_API_KEY or pass --openai-key")
        return

    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    # Load questions
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        return

    questions = load_questions(str(questions_path))
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Connect to ChromaDB
    collection, success, error = rag_client.initialize_rag_system(
        args.chroma_dir, args.collection_name
    )
    if not success:
        print(f"ERROR: Failed to connect to ChromaDB: {error}")
        return

    print(f"Connected to collection '{args.collection_name}'")

    # Run batch evaluation
    results = run_batch_evaluation(
        questions, collection, args.openai_key,
        model=args.model, num_docs=args.num_docs
    )

    # Print summary table
    print_summary_table(results)


if __name__ == "__main__":
    main()
