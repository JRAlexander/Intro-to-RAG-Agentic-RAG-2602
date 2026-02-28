from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path
    DATA_RAW_NOTES_DIR: Path
    DATA_PROCESSED_DIR: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
config = Config(
    PROJECT_ROOT=PROJECT_ROOT,
    DATA_RAW_NOTES_DIR=PROJECT_ROOT / "data" / "raw" / "notes",
    DATA_PROCESSED_DIR=PROJECT_ROOT / "data" / "processed",
)
