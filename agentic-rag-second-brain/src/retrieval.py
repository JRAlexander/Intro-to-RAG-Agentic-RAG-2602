from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.index_store import COLLECTION_NAME


def load_persisted_index(chroma_dir: Path | str, embed_model: str) -> VectorStoreIndex:
    chroma_path = Path(chroma_dir)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        raise FileNotFoundError(
            f"Persisted Chroma directory not found or empty: {chroma_path}. "
            "Run notebooks/02_indexing_chroma_llamaindex.ipynb first."
        )

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        raise FileNotFoundError(
            f"Chroma collection '{COLLECTION_NAME}' has no vectors at {chroma_path}. "
            "Run notebooks/02_indexing_chroma_llamaindex.ipynb first."
        )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    embedding = OpenAIEmbedding(model=embed_model)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding)


def retrieve_chunks(index: VectorStoreIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    rows: list[dict[str, Any]] = []
    for result in results:
        node = result.node
        rows.append(
            {
                "score": float(result.score) if result.score is not None else None,
                "text": node.get_content(),
                "doc_title": node.metadata.get("doc_title", ""),
                "doc_date": node.metadata.get("doc_date", ""),
                "chunk_id": node.metadata.get("chunk_id", ""),
                "source_path": node.metadata.get("source_path", ""),
            }
        )

    return rows
