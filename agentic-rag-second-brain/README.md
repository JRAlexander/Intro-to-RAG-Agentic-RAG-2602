# Agentic RAG Second Brain (Setup + Dataset + Ingestion)

This module intentionally scopes to:
- environment and project setup,
- dataset preparation/loading,
- ingestion pipeline construction.

It **does not** include retrieval, vector store wiring, or LangGraph orchestration.

## Project structure

```text
agentic-rag-second-brain/
├── pyproject.toml
├── environment.yml
├── .env.example
├── README.md
├── data/
│   └── raw/
│       └── notes/
├── notebooks/
│   ├── 00_setup.ipynb
│   └── 01_dataset_and_ingestion_llamaindex.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    └── ingestion.py
```

## Quick start

1. Create environment:
   - Conda: `conda env create -f environment.yml`
   - Or pip/uv: `pip install -e .`
2. Copy env template:
   - `cp .env.example .env`
3. Add markdown notes into `data/raw/notes/`.
4. Run notebooks in order:
   - `notebooks/00_setup.ipynb`
   - `notebooks/01_dataset_and_ingestion_llamaindex.ipynb`
