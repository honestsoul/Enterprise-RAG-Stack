#!/usr/bin/env python3
"""Build embeddings for a directory of text files and persist them.

This script reads all ``.txt`` files in the specified input directory,
cleans and chunks them, then computes embeddings using the model
configured as the default provider.  The resulting FAISS index and
chunk mapping are written to disk for later use by the inference
engine.

Usage:

    python build_embeddings.py --input-dir ./mydocs --output-dir ./data/vectordb

The output directory will contain two files:

* ``index.faiss`` – the binary FAISS index
* ``doc_chunks.json`` – a JSON mapping of chunk identifiers to text
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from generative_ai_project.core.model_factory import get_default_model
from generative_ai_project.processing.preprocess import preprocess_documents
from generative_ai_project.rag.indexer import Indexer
from generative_ai_project.rag.vector_store import VectorStore


def load_documents(input_dir: Path) -> Dict[str, str]:
    """Load all text files from a directory into a mapping.

    Args:
        input_dir: Directory containing ``.txt`` files.

    Returns:
        A dictionary mapping file stems to their contents.
    """
    documents: Dict[str, str] = {}
    for path in input_dir.glob("*.txt"):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document embeddings and FAISS index.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .txt files to index")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the index and metadata")
    args = parser.parse_args()

    documents = load_documents(args.input_dir)
    if not documents:
        print(f"No .txt files found in {args.input_dir}")
        return

    # Preprocess documents into chunks
    processed = preprocess_documents(documents)
    # Flatten the chunk mapping for indexing and save for later retrieval
    doc_chunks: Dict[str, str] = {}
    for doc_id, chunks in processed.items():
        for i, chunk in enumerate(chunks):
            doc_chunks[f"{doc_id}#{i}"] = chunk

    # Determine the dimension for the vector store by computing one embedding
    model = get_default_model()
    sample_embedding = model.embed([next(iter(doc_chunks.values()))])[0]
    store = VectorStore(dimension=len(sample_embedding))
    # Index all documents
    indexer = Indexer(store, model)
    indexer.index_documents(documents)

    # Write the FAISS index and chunk mapping to disk
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = args.output_dir / "index.faiss"
    doc_path = args.output_dir / "doc_chunks.json"
    import faiss  # type: ignore

    faiss.write_index(store.index, str(faiss_path))
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(doc_chunks, f)
    print(f"Wrote FAISS index to {faiss_path} and chunk metadata to {doc_path}")


if __name__ == "__main__":
    main()
