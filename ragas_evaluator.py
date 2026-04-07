import os
import asyncio
from typing import Dict, List

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Score the response against RAGAS faithfulness, relevancy, BLEU, and ROUGE metrics."""

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available. Install with: pip install ragas"}

    if not question or not answer:
        return {"error": "Question and answer must not be empty"}

    if not contexts:
        contexts = [""]

    try:
        api_key = os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

        scoring_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                base_url="https://openai.vocareum.com/v1"
            )
        )

        scoring_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
                base_url="https://openai.vocareum.com/v1"
            )
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        evaluation_metrics = {
            "faithfulness": Faithfulness(llm=scoring_llm),
            "answer_relevancy": ResponseRelevancy(
                llm=scoring_llm,
                embeddings=scoring_embeddings
            ),
            "bleu_score": BleuScore(),
            "rouge_score": RougeScore(),
        }

        scores = {}
        for metric_name, metric in evaluation_metrics.items():
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        score = loop.run_until_complete(metric.single_turn_ascore(sample))
                    else:
                        score = asyncio.run(metric.single_turn_ascore(sample))
                except RuntimeError:
                    score = asyncio.run(metric.single_turn_ascore(sample))

                scores[metric_name] = round(float(score), 4)

            except Exception as e:
                scores[metric_name] = 0.0

        return scores

    except Exception as e:
        return {"error": str(e)}
