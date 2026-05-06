import json
import logging
from typing import Type

import chromadb
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from draftr.vault_config import SentenceTransformerEF, load_vault_config

log = logging.getLogger(__name__)


class KnowledgeVaultInput(BaseModel):
    requirement_id: str = Field(..., description="Requirement ID, e.g. REQ-001.")
    requirement_text: str = Field(..., description="Full requirement text to search against.")
    top_k: int = Field(default=5, description="Number of results to return.")


class KnowledgeVaultTool(BaseTool):
    name: str = "knowledge_vault_search"
    description: str = (
        "Searches the company Knowledge Vault of past winning bids for content "
        "relevant to a given RFP requirement. Returns ranked snippets with source "
        "document and relevance score. Always call this before writing any section "
        "from scratch — reuse proven language whenever a strong match is found."
    )
    args_schema: Type[BaseModel] = KnowledgeVaultInput

    def _run(self, requirement_id: str, requirement_text: str, top_k: int = 5) -> str:
        vc = load_vault_config()

        ef = SentenceTransformerEF(model_name=vc.embedding_model)

        try:
            client = chromadb.HttpClient(host=vc.host, port=vc.port)
        except Exception as exc:
            log.warning("ChromaDB unavailable at %s:%s — %s", vc.host, vc.port, exc)
            return json.dumps(
                self._not_ready(requirement_id, f"Cannot connect to ChromaDB at {vc.host}:{vc.port} — {exc}")
            )

        try:
            collection = client.get_collection(name=vc.collection, embedding_function=ef)
        except Exception:
            log.warning("Collection '%s' not found on %s:%s", vc.collection, vc.host, vc.port)
            return json.dumps(
                self._not_ready(
                    requirement_id,
                    f"Collection '{vc.collection}' not found on {vc.host}:{vc.port}.",
                )
            )

        count = collection.count()
        if count == 0:
            log.warning("Vault collection '%s' is empty — run ingest_bids first.", vc.collection)
            return json.dumps(self._not_ready(requirement_id, "Vault collection is empty."))

        effective_k = min(top_k or vc.top_k, count)
        results = collection.query(
            query_texts=[requirement_text],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        snippets = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine space: distance 0 = identical, 1 = orthogonal → similarity = 1 − distance.
            relevance_score = round(1.0 - float(dist), 4)
            snippets.append(
                {
                    "source_document": meta.get("source_file", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "chunk_index": meta.get("chunk_index", 0),
                    "relevance_score": relevance_score,
                    "snippet_text": doc,
                }
            )

        snippets.sort(key=lambda x: x["relevance_score"], reverse=True)
        knowledge_gap = all(s["relevance_score"] < vc.min_relevance_score for s in snippets)

        return json.dumps(
            {"requirement_id": requirement_id, "knowledge_gap": knowledge_gap, "results": snippets},
            indent=2,
        )

    @staticmethod
    def _not_ready(requirement_id: str, reason: str) -> dict:
        return {
            "requirement_id": requirement_id,
            "knowledge_gap": True,
            "results": [],
            "warning": f"{reason} Run: ingest_bids  (or: python scripts/ingest_bids.py)",
        }
