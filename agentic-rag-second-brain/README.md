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
# agentic-rag-second-brain

Minimal setup for **Notebook 00 (setup)** and **Notebook 01 (dataset + ingestion with LlamaIndex)**.

## Setup A: pip (default)

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```
4. Install project dependencies:
   ```bash
   pip install -e .
   ```
5. Register kernel:
   ```bash
   python -m ipykernel install --user --name agentic-rag-second-brain
   ```
6. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
7. Run notebooks in order:
   - `notebooks/00_setup.ipynb`
   - `notebooks/01_dataset_and_ingestion_llamaindex.ipynb`

## Setup B: conda (optional)

1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate agentic-rag-second-brain
   ```
3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
4. Run notebooks in order:
   - `notebooks/00_setup.ipynb`
   - `notebooks/01_dataset_and_ingestion_llamaindex.ipynb`

## Acceptance check

```bash
python -c "from src.dataset import ensure_dataset_exists; ensure_dataset_exists()"
```

## Notebook 02 prerequisites (Chroma indexing + persistence)

Before running `notebooks/02_indexing_chroma_llamaindex.ipynb`, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key"
```

Run notebooks in this order:
1. `notebooks/00_setup.ipynb`
2. `notebooks/01_dataset_and_ingestion_llamaindex.ipynb`
3. `notebooks/02_indexing_chroma_llamaindex.ipynb`

To force a clean rebuild of the persisted Chroma index, set:

```bash
export RESET_INDEX=1
```

Use `RESET_INDEX=0` (default) to reuse the existing persisted index for quicker reruns.


## Notebook 05: Evaluation workflow

This repo includes a lightweight, repeatable eval harness for comparing baseline and agentic RAG.

### Artifacts
- Golden dataset: `eval/golden_questions.jsonl`
- Notebook: `notebooks/05_eval.ipynb`
- Eval helpers: `src/eval.py` (and optional `src/eval_judge.py`)

### Run
1. Ensure the persisted index exists (`notebooks/02_indexing_chroma_llamaindex.ipynb`).
2. Run baseline + agentic notebooks as needed (`03`, `04`) to validate your setup.
3. Open and run `notebooks/05_eval.ipynb`.

The notebook reports baseline vs agentic metrics for:
- citation presence rate
- citation validity rate (chunk IDs exist)
- recency correctness rate (drift subset only)
- average retries (agentic)
- average latency per question

It also prints a short top-failures section (3 examples) with query, retrieved doc titles/dates, answer, citations, and failed checks.

### Optional LLM-as-judge
By default, LLM judging is disabled.

Set in `.env`:
```bash
USE_LLM_EVAL=1
```
When enabled, `run_eval` appends `llm_judge_score` and `llm_judge_rationale` columns via `src/eval_judge.py`.
