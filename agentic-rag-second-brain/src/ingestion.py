from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def _parse_frontmatter(text: str) -> Dict[str, object]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError("Markdown note missing YAML frontmatter block.")

    frontmatter_lines: List[str] = []
    body_start = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            body_start = idx + 1
            break
        frontmatter_lines.append(lines[idx])

    if body_start is None:
        raise ValueError("Unterminated YAML frontmatter block.")

    parsed: Dict[str, object] = {}
    i = 0
    while i < len(frontmatter_lines):
        line = frontmatter_lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith("tags:"):
            tags: List[str] = []
            i += 1
            while i < len(frontmatter_lines) and frontmatter_lines[i].lstrip().startswith("-"):
                tags.append(frontmatter_lines[i].split("-", 1)[1].strip())
                i += 1
            parsed["tags"] = tags
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        i += 1

    parsed["body"] = "\n".join(lines[body_start:]).strip()
    return parsed


def _doc_id(source_path: str, title: str, date: str) -> str:
    payload = f"{source_path}|{title}|{date}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_markdown_documents(notes_dir) -> List[Document]:
    notes_path = Path(notes_dir)
    documents: List[Document] = []

    for path in sorted(notes_path.glob("*.md")):
        parsed = _parse_frontmatter(path.read_text(encoding="utf-8"))
        title = str(parsed.get("title", "Untitled"))
        date = str(parsed.get("date", ""))
        tags = parsed.get("tags", [])
        source_path = str(path.resolve())
        doc_id = _doc_id(source_path=source_path, title=title, date=date)

        metadata = {
            "doc_title": title,
            "doc_date": date,
            "tags": tags,
            "source_path": source_path,
            "doc_id": doc_id,
        }

        documents.append(
            Document(
                text=str(parsed.get("body", "")),
                metadata=metadata,
                id_=doc_id,
            )
        )

    return documents


def chunk_documents(documents) -> List:
    parser = SentenceSplitter(chunk_size=420, chunk_overlap=60)
    nodes = parser.get_nodes_from_documents(documents)

    for index, node in enumerate(nodes):
        source_doc_id = node.metadata.get("doc_id", node.ref_doc_id or "unknown")
        node.metadata["chunk_id"] = f"{source_doc_id}:{index}"

    return nodes
