# Draftr — Runbook

Step-by-step guide from zero to a finished proposal.

> Full architecture and design documentation is in `architecture.docx` (internal, not tracked in git).

---

## Step 1 — Install dependencies

```bash
cd crew_ai/draftr
uv sync
# or: pip install -e .
```

---

## Step 2 — Set API keys

Create a `.env` file in `crew_ai/draftr/` and add:

```
OPENAI_API_KEY=sk-...          # Requirements Architect + Knowledge Retriever
GEMINI_API_KEY=AIza...         # Technical Writer
ANTHROPIC_API_KEY=sk-ant-...   # Compliance Auditor
```

---

## Step 3 — Start ChromaDB

Draftr connects to ChromaDB over HTTP. Start it before ingesting or running:

```bash
# pip
pip install chromadb
chroma run --port 8170 --path ./chroma_data

# Docker
docker run -p 8170:8000 -v $(pwd)/chroma_data:/chroma/chroma chromadb/chroma
```

Verify `vault.host` and `vault.port` in `config.yaml` match.

---

## Step 4 — Add past winning bids

Drop your past winning proposal documents into:

```
knowledge/past_bids/
```

Supported formats: `.pdf` `.docx` `.txt` `.md` — subdirectories are scanned recursively.
These files are gitignored and will never be accidentally committed.

---

## Step 5 — Ingest bids into the Knowledge Vault

```bash
ingest_bids                          # incremental (upserts only new chunks)
ingest_bids --reset                  # wipe and rebuild from scratch
ingest_bids --bids-dir /path/to/bids # override source directory
```

Expected output:

```
[info] Connecting to ChromaDB at localhost:8170
[info] Embedding model : all-MiniLM-L6-v2
[info] Collection      : proposal_mine

  Processing: winning_bid_acme_2024.pdf
    → 42 chunk(s) ingested

[done] 42 total chunks from 1 file(s)
```

---

## Step 6 — Configure and run

**Option A — CLI flags:**

```bash
draftr \
  --rfp-path /path/to/rfp.pdf \
  --client-name "Acme Corporation" \
  --rfp-tone "formal government"
```

**Option B — config.yaml:**

```yaml
rfp:
  path: "path/to/rfp.pdf"
  client_name: "Acme Corporation"
  tone: "formal government"
```

```bash
draftr
```

**Validate inputs without running the crew:**

```bash
draftr --rfp-path rfp.pdf --client-name "Acme" --dry-run
```

**Log verbosity:**

```bash
draftr --rfp-path rfp.pdf --client-name "Acme" --log-level DEBUG
```

---

## Step 7 — Review outputs

All outputs are written to `output/` (configurable via `output.dir` in `config.yaml`):

| File | Produced by | What to look for |
|---|---|---|
| `requirements.json` | Requirements Architect | Confirm every RFP clause was captured |
| `knowledge.json` | Knowledge Retriever | Check `knowledge_gaps` — these sections were drafted from scratch |
| `proposal.md` | Technical Writer | First-draft proposal — review and refine before submission |
| `compliance.md` | Compliance Auditor | **Gap Report** → revise and re-run · **Compliance Stamp** → ready to submit |

If `compliance.md` is a Gap Report, address the flagged items in `proposal.md` and re-run from Step 6.

---

## Other entry points

```bash
# Train the crew
train --rfp-path rfp.pdf --client-name "Acme" 5 training_results.json

# Evaluate
test --rfp-path rfp.pdf --client-name "Acme" 3 gpt-4o

# Replay from a specific task (useful for debugging mid-pipeline)
replay <task_id>

# Remote scheduler trigger (JSON payload via argv or stdin)
run_with_trigger '{"rfp_path": "...", "client_name": "...", "rfp_tone": "..."}'
cat payload.json | run_with_trigger
```

---

## Configuration reference

All parameters live in `config.yaml`. CLI flags always take priority over config file values.

| Key | Default | Description |
|---|---|---|
| `rfp.path` | `""` | Path to the RFP document (required) |
| `rfp.client_name` | `""` | Client / issuing organisation name (required) |
| `rfp.tone` | `"formal government"` | Tone directive for the Technical Writer |
| `output.dir` | `"output"` | Directory for all output files |
| `vault.host` | `"localhost"` | ChromaDB HTTP server hostname |
| `vault.port` | `8170` | ChromaDB HTTP server port |
| `vault.collection` | `"proposal_mine"` | ChromaDB collection name |
| `embedding.model` | `"all-MiniLM-L6-v2"` | Sentence-transformer model (must match between ingest and retrieval) |
| `embedding.min_relevance_score` | `0.75` | Cosine similarity threshold — below this flags a knowledge gap |
| `embedding.top_k` | `5` | Max snippets returned per requirement |
| `ingestion.bids_dir` | `"knowledge/past_bids"` | Source directory for bid documents |
| `ingestion.chunk_size` | `1000` | Characters per text chunk |
| `ingestion.overlap` | `150` | Character overlap between consecutive chunks |
