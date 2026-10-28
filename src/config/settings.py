"""Centralized settings for the offline corpus-building pipeline."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base directory (project root)
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    # Data paths
    DATA_RAW_DIR: Path = Field(default=Path("data/raw"))
    DATA_PARSED_DIR: Path = Field(default=Path("data/parsed"))
    DATA_CHUNKS_DIR: Path = Field(default=Path("data/chunks"))

    # Vector DB
    CHROMA_PERSIST_DIR: Path = Field(default=Path("vector_db/chroma"))
    LEGAL_COLLECTION_NAME: str = "legal_corpus"
    SCIENTIFIC_COLLECTION_NAME: str = "scientific_corpus"

    # Embedding configurations
    # Scientific model: SPECTER2 default
    SCIENTIFIC_EMBEDDING_MODEL: str = "allenai/specter2_base"
    # Legal model: BGE-large default
    LEGAL_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    EMBEDDING_NORMALIZE: bool = True
    DATA_EMBEDDINGS_DIR: Path = Field(default=Path("data/embeddings"))

    # Chunking configurations
    SCIENTIFIC_MAX_CHUNK_TOKENS: int = 512
    SCIENTIFIC_CHUNK_OVERLAP: int = 64
    LEGAL_MAX_CHUNK_TOKENS: int = 512

    # Logging
    LOG_LEVEL: str = "INFO"

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative path against BASE_DIR."""
        if path.is_absolute():
            return path
        return (self.BASE_DIR / path).resolve()

    @property
    def raw_legal_dir(self) -> Path:
        return self.resolve_path(self.DATA_RAW_DIR / "legal")

    @property
    def raw_scientific_dir(self) -> Path:
        return self.resolve_path(self.DATA_RAW_DIR / "scientific")

    @property
    def parsed_legal_dir(self) -> Path:
        return self.resolve_path(self.DATA_PARSED_DIR / "legal")

    @property
    def parsed_scientific_dir(self) -> Path:
        return self.resolve_path(self.DATA_PARSED_DIR / "scientific")

    @property
    def chunks_legal_dir(self) -> Path:
        return self.resolve_path(self.DATA_CHUNKS_DIR / "legal")

    @property
    def chunks_scientific_dir(self) -> Path:
        return self.resolve_path(self.DATA_CHUNKS_DIR / "scientific")

    @property
    def embeddings_legal_dir(self) -> Path:
        return self.resolve_path(self.DATA_EMBEDDINGS_DIR / "legal")

    @property
    def embeddings_scientific_dir(self) -> Path:
        return self.resolve_path(self.DATA_EMBEDDINGS_DIR / "scientific")

    @property
    def chroma_dir(self) -> Path:
        return self.resolve_path(self.CHROMA_PERSIST_DIR)

    def ensure_directories(self) -> None:
        """Ensure all required pipeline directories exist on disk."""
        dirs = [
            self.raw_legal_dir,
            self.raw_scientific_dir,
            self.parsed_legal_dir,
            self.parsed_scientific_dir,
            self.chunks_legal_dir,
            self.chunks_scientific_dir,
            self.embeddings_legal_dir,
            self.embeddings_scientific_dir,
            self.chroma_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    settings = Settings()
    return settings
