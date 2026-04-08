#!/usr/bin/env python3
"""
NASA Mission Intelligence Chat
Streamlit interface for querying NASA mission documents via RAG,
with integrated RAGAS evaluation for response quality tracking.
"""

import streamlit as st
import os
import json
import pandas as pd

import ragas_evaluator
import rag_client
import llm_client

from pathlib import Path
from typing import Dict, List, Optional

try:
    from ragas import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    st.warning("RAGAS not available. Install with: pip install ragas")

st.set_page_config(
    page_title="NASA Mission Intelligence",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Return all available ChromaDB backends found in the project directory."""
    return rag_client.discover_chroma_backends()

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Connect to the selected ChromaDB collection."""
    try:
       return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, num_docs: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Fetch the most relevant document chunks for the given query."""
    try:
        return rag_client.retrieve_documents(collection, query, num_docs, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Assemble retrieved chunks into a formatted context string."""
    return rag_client.format_context(documents, metadatas)

def generate_response(openai_key, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Send the user message and context to OpenAI and return the answer."""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Run RAGAS evaluation metrics on the latest response."""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def display_evaluation_metrics(scores: Dict[str, float]):
    """Render RAGAS quality scores in the sidebar."""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            if score >= 0.8:
                color = "green"
            elif score >= 0.6:
                color = "orange"
            else:
                color = "red"

            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}",
                delta=None
            )
            st.sidebar.progress(score)

def main():
    st.title("🚀 NASA Mission Intelligence Chat")
    st.markdown("Ask questions about NASA space missions — answers are grounded in official mission documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n`python embedding_pipeline.py`")
            st.stop()

        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )

        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )

        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        else:
            os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the OpenAI model for responses"
        )

        st.subheader("🔍 Retrieval Settings")
        num_docs = st.slider("Documents to retrieve", 1, 10, 3)

        st.subheader("📊 Evaluation Settings")
        eval_enabled = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)

        if st.session_state.current_backend != selected_backend_key:
            st.session_state.current_backend = selected_backend_key
            st.cache_resource.clear()

    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    if st.session_state.last_evaluation and eval_enabled:
        display_evaluation_metrics(st.session_state.last_evaluation)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                retrieval_result = retrieve_documents(collection, prompt, num_docs)

                context = ""
                retrieved_chunks = []
                if retrieval_result and retrieval_result.get("documents"):
                    context = format_context(
                        retrieval_result["documents"][0],
                        retrieval_result["metadatas"][0]
                    )
                    retrieved_chunks = retrieval_result["documents"][0]
                    st.session_state.last_contexts = retrieved_chunks

                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
                st.markdown(response)

                if eval_enabled and RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        quality_metrics = evaluate_response_quality(
                            prompt,
                            response,
                            retrieved_chunks
                        )
                        st.session_state.last_evaluation = quality_metrics

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
