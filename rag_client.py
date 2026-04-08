import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Scan the project directory and return all available ChromaDB backends."""
    available_stores = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and ('chroma' in d.name.lower() or 'db' in d.name.lower())
        and not d.name.startswith('.')
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"

                try:
                    count = collection.count()
                except Exception:
                    count = 0

                available_stores[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({count} docs) — {chroma_dir.name}",
                }

        except Exception as e:
            available_stores[str(chroma_dir)] = {
                "directory": str(chroma_dir),
                "collection_name": "unknown",
                "display_name": f"{chroma_dir.name} — Error: {str(e)[:40]}",
            }

    return available_stores


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Connect to an existing ChromaDB collection and return it."""
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Query ChromaDB for the most relevant document chunks, with optional mission scoping."""

    query_filter = None

    if mission_filter and mission_filter.strip().lower() not in ["all", "none", "", "any"]:
        query_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=query_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Build a readable context block from retrieved chunks and their metadata."""
    if not documents:
        return ""

    formatted_sections = ["=== Retrieved Context from NASA Documents ===\n"]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        mission = metadata.get('mission', 'unknown').replace('_', ' ').title()
        source = metadata.get('source', 'unknown')
        category = metadata.get('document_category', 'unknown').replace('_', ' ').title()

        header = f"[Source {i + 1}] Mission: {mission} | File: {source} | Category: {category}"
        formatted_sections.append(header)
        formatted_sections.append("-" * len(header))

        if len(doc) > 800:
            doc = doc[:800] + "... [truncated]"

        formatted_sections.append(doc)
        formatted_sections.append("")

    return "\n".join(formatted_sections)
