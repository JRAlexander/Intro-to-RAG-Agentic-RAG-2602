from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

COLLECTION_NAME = "notes"


def _has_persisted_index(chroma_dir: Path, collection) -> bool:
    """Return True when persisted files and vectors both exist."""

    return chroma_dir.exists() and any(chroma_dir.iterdir()) and collection.count() > 0


def build_or_load_index(nodes: Sequence, reset: bool, chroma_dir: Path, embed_model: str):
    """Build or load a persisted Chroma-backed vector index.

    - If ``reset`` is True, any existing persisted Chroma directory is removed.
    - If data already exists and ``reset`` is False, the existing index is loaded.
    - Otherwise, a new index is built from ``nodes`` and persisted to ``chroma_dir``.
    """

    chroma_dir = Path(chroma_dir)

    if reset and chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    chroma_dir.mkdir(parents=True, exist_ok=True)

    embed = OpenAIEmbedding(model=embed_model)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if _has_persisted_index(chroma_dir=chroma_dir, collection=chroma_collection) and not reset:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed,
        )
        built = False
    else:
        index = VectorStoreIndex(
            nodes=list(nodes),
            storage_context=storage_context,
            embed_model=embed,
        )
        storage_context.persist(persist_dir=str(chroma_dir))
        built = True

    return {
        "index": index,
        "built": built,
        "collection_name": COLLECTION_NAME,
        "chroma_dir": chroma_dir,
        "vector_count": chroma_collection.count(),
    }
