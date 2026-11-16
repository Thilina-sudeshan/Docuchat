"""Application settings, loaded from environment variables / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    claude_effort: str = os.getenv("CLAUDE_EFFORT", "medium")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))
    collection_name: str = os.getenv("COLLECTION_NAME", "docuchat")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))


settings = Settings()
