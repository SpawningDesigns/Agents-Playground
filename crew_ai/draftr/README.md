# Draftr

Multi-agent AI pipeline that reads an RFP, searches past winning bids, and produces a compliance-verified proposal draft in one automated run.

Built with [CrewAI](https://docs.crewai.com/) · ChromaDB · Sentence-Transformers · OpenAI · Google Gemini · Anthropic Claude.

## Architecture

Full design documentation is in `docs/architecture.docx` (internal, not tracked in git).
It covers the high-level design, agent pipeline, Knowledge Vault internals, low-level module design, data models, configuration schema, multi-LLM strategy, and extension points.

## Quick Start

See `docs/README.md` for the complete step-by-step runbook.

```bash
# 1. Install
cd crew_ai/draftr
uv sync

# 2. Set API keys in .env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...

# 3. Start ChromaDB
chroma run --port 8170 --path ./chroma_data

# 4. Ingest past winning bids
#    Drop .pdf/.docx/.txt/.md files into knowledge/past_bids/, then:
ingest_bids

# 5. Run
draftr --rfp-path path/to/rfp.pdf --client-name "Acme Corp" --rfp-tone "formal government"
```

Outputs are written to `output/`: `requirements.json`, `knowledge.json`, `proposal.md`, `compliance.md`.
