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
