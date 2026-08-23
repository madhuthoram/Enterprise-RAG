#!/usr/bin/env python
"""
Basic example showing how to query the Enterprise-Ready RAG System using the Python API.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.enterprise_rag import RAGSystem


def main():
    """Run a basic query example."""

    # Initialize the RAG system with a basic configuration
    rag_system = RAGSystem(
        vector_store_config={
            "type": "faiss",
            "index_path": "data/index",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        },
        retrieval_config={"type": "hybrid", "top_k": 5, "use_reranker": True},
        generation_config={
            # Free OpenRouter model (zero credits needed) — requires
            # OPENAI_BASE_URL=https://openrouter.ai/api/v1. Free-tier slugs
            # rotate; re-check https://openrouter.ai/models?max_price=0 if this 404s.
            "model": "nvidia/nemotron-3.5-lightning:free",
            "temperature": 0.7,
            "max_tokens": 500,
        },
    )

    # Define your query
    query = (
        "What are the key benefits of using a RAG system for enterprise applications?"
    )

    # Optional query parameters
    # Note: RAGSystem.query() takes filters/retrieval_options/generation_options
    # as separate keyword args, not a single bundled "options" dict.
    # No metadata filters here — the sample doc from document_indexing_example.py
    # has no "doc_type" field, so a doc_type="pdf" filter would exclude it.
    retrieval_options = {"top_k": 5}

    # Execute the query
    print(f"Querying: '{query}'")

    response = rag_system.query(query, retrieval_options=retrieval_options)

    # Print the results
    print("\n----- ANSWER -----")
    print(response["answer"])

    print("\n----- SOURCES -----")
    if not response["sources"]:
        print("(none cited)")
    for i, source in enumerate(response["sources"], 1):
        label = source.get("title") or source.get("filename") or source.get("id")
        print(f"{i}. {label} (confidence: {source.get('confidence', 0):.2f})")

    # Print performance metrics
    print("\n----- PERFORMANCE METRICS -----")
    metadata = response["metadata"]
    print(f"retrieved_doc_count: {metadata['retrieved_doc_count']}")
    print(f"confidence: {metadata['confidence']:.3f}")
    for key, value in metadata["latency"].items():
        print(f"{key}: {value / 1000:.3f}s")


if __name__ == "__main__":
    # Check if API key is set (OPENAI_API_KEY also carries an OpenRouter key
    # when OPENAI_BASE_URL is set to https://openrouter.ai/api/v1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY=your_api_key_here")
        print("Using OpenRouter? Also run: export OPENAI_BASE_URL=https://openrouter.ai/api/v1")
        print("For this example, using a mock response instead.\n")

    main()
