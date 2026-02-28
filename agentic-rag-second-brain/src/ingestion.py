from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter


def build_ingestion_pipeline(chunk_size: int = 512, chunk_overlap: int = 64) -> IngestionPipeline:
    """Build a basic ingestion pipeline that chunks documents into nodes."""

    transformations = [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    ]
    return IngestionPipeline(transformations=transformations)
