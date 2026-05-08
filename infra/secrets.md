# Secrets

All secrets are read from environment variables. `.env.example` is the canonical
list — copy it to `.env` for local development. Never commit `.env`.

| Variable | Where it's used | Where to set in production |
|---|---|---|
| `HF_TOKEN` | Pushing snapshots, model artifacts, Space deploys; pulling private artifacts | GitHub Actions secret + HF Space secret |
| `DISCORD_WEBHOOK_URL` | `monitoring/alerts.py` — ingest, drift, retrain pipeline notifications | GitHub Actions secret |
| `MLFLOW_TRACKING_URI` | Local + CI MLflow runs (default: `sqlite:///mlruns/mlflow.db`) | Set per-environment if remote tracking is added |
| `QDRANT_PATH` | Vector store local-mode storage path (default: `./qdrant_storage`; on Spaces: `/data/qdrant`) | Configured automatically on Spaces |

## Configuring secrets

### Local development
```sh
cp .env.example .env
# fill in HF_TOKEN and DISCORD_WEBHOOK_URL
```

### GitHub Actions
```sh
gh secret set HF_TOKEN
gh secret set DISCORD_WEBHOOK_URL
```

### Hugging Face Space
Settings → Variables and secrets → New secret → add `HF_TOKEN` and
`DISCORD_WEBHOOK_URL`. Spaces auto-restart on secret changes.

## Token scopes

`HF_TOKEN` needs **write** access to:
- Dataset repo `arjun10g/na-tech-jobs`
- Model repos `arjun10g/na-tech-jobs-*`
- Space repo `arjun10g/na-tech-jobs`

Generate at https://huggingface.co/settings/tokens with the "Write" preset.
