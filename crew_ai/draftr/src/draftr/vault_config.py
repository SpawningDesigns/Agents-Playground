from dataclasses import dataclass
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


@dataclass
class VaultConfig:
    # ChromaDB HTTP connection
    host: str
    port: int
    collection: str
    # Embedding
    embedding_model: str
    min_relevance_score: float
    top_k: int
    # Ingestion
    chunk_size: int
    overlap: int
    bids_dir: Path


def load_vault_config(config_path: Path = DEFAULT_CONFIG) -> VaultConfig:
    """Load vault-related settings from config.yaml with safe built-in defaults."""
    cfg: dict = {}
    if config_path.exists():
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}

    vault = cfg.get("vault", {})
    embedding = cfg.get("embedding", {})
    ingestion = cfg.get("ingestion", {})

    bids_dir_raw = ingestion.get("bids_dir", "knowledge/past_bids")
    bids_dir = Path(bids_dir_raw)
    if not bids_dir.is_absolute():
        bids_dir = _PROJECT_ROOT / bids_dir

    return VaultConfig(
        host=str(vault.get("host", "localhost")),
        port=int(vault.get("port", 8170)),
        collection=str(vault.get("collection", "proposal_mine")),
        embedding_model=str(embedding.get("model", "all-MiniLM-L6-v2")),
        min_relevance_score=float(embedding.get("min_relevance_score", 0.75)),
        top_k=int(embedding.get("top_k", 5)),
        chunk_size=int(ingestion.get("chunk_size", 1000)),
        overlap=int(ingestion.get("overlap", 150)),
        bids_dir=bids_dir,
    )


class SentenceTransformerEF:
    """Embedding function that wraps sentence-transformers for use with chromadb-client.

    chromadb-client (thin HTTP client) ships no embedding implementations.
    This class owns that responsibility so both the ingestion pipeline and
    the retrieval tool use an identical, single-sourced implementation.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._model.encode(input, convert_to_numpy=True).tolist()
