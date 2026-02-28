import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""

    raw_notes_dir: str = os.getenv("RAW_NOTES_DIR", "data/raw/notes")
    default_source_tag: str = os.getenv("DEFAULT_SOURCE_TAG", "second-brain")


settings = Settings()
