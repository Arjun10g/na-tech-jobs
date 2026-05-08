# na-tech-jobs

A production ML platform for the **North American senior tech-hiring market**:
weekly ATS ingestion across Greenhouse, Lever, Ashby, Workable, SmartRecruiters,
and curated Workday tenants → a versioned dataset on the Hugging Face Hub →
a multi-model pipeline (salary regression, seniority and role-family classifiers,
skill extraction, embeddings) → a hybrid + late-interaction RAG layer with
LLM-powered retrieval and natural-language analytics → all deployed on a $9
Hugging Face Pro Space with ZeroGPU, drift-monitored weekly and retrained
monthly.

> Built by a senior data science candidate using the platform for their own
> North American job search.

The full project bible — architecture, model choices, phased plan, risks,
out-of-scope decisions — lives in [`CLAUDE.md`](CLAUDE.md).

## Status

✅ **Phase 0 — scaffolding.** Repo bootstrapped, CI green, hello-world Gradio
app live at https://arjun10g-na-tech-jobs.hf.space.

🚧 **Phase 1 — ingestion v1.** Greenhouse + Lever + Ashby extractors,
async-fan-out orchestrator, Pandera-validated parquet snapshots pushed weekly
to https://huggingface.co/datasets/arjun10g/na-tech-jobs. Latest snapshot:
~12.3k jobs across 65 verified boards (US 95% / CA 5%).

See [`CLAUDE.md` § 10](CLAUDE.md) for the full phased plan.

## Quickstart

```sh
# install Python 3.11 + dependencies
uv sync --group dev

# run the local Gradio app
uv run python -m app.main

# run lint + tests
uv run ruff format --check .
uv run ruff check .
uv run pytest

# run a smoke ingest (5 companies, no HF push)
uv run python -m ingestion.orchestrator --output-dir data --limit 5

# full weekly ingest with HF Dataset push (needs HF_TOKEN)
uv run python -m ingestion.orchestrator --output-dir data --push-to-hub --alert
```

Optional install groups:

| Group | Purpose | Install |
|---|---|---|
| `ml` | Training stack (torch, transformers, xgboost, mlflow, …) | `uv sync --extra ml` |
| `rag` | Vector store, chunking, NL→SQL, PDF parsing | `uv sync --extra rag` |
| `monitoring` | Evidently for drift reports | `uv sync --extra monitoring` |
| `api` | FastAPI for programmatic endpoints | `uv sync --extra api` |

## Repository layout

See [`CLAUDE.md` § 9](CLAUDE.md). Top-level packages are created phase-by-phase;
Phase 0 ships only `app/` plus configuration.

## Secrets

See [`infra/secrets.md`](infra/secrets.md). Copy `.env.example` to `.env` and
fill in `HF_TOKEN` + `DISCORD_WEBHOOK_URL` to run anything that touches the Hub
or alerting.

## License

MIT. Models and datasets are licensed individually — see their respective
model and dataset cards.
