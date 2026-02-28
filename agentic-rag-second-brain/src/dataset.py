from pathlib import Path

from llama_index.core import Document

from .config import settings


def load_note_documents(raw_notes_dir: str | None = None) -> list[Document]:
    """Load markdown/text notes into LlamaIndex Document objects."""

    notes_path = Path(raw_notes_dir or settings.raw_notes_dir)
    if not notes_path.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(notes_path.glob("**/*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                text=text,
                metadata={
                    "source": settings.default_source_tag,
                    "path": str(file_path),
                    "filename": file_path.name,
                },
            )
        )

    return documents
