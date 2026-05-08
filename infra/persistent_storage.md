# Persistent storage on Hugging Face Spaces

The Space is configured with **20 GB persistent storage** mounted at `/data`.
Anything outside `/data` is wiped on container restart.

## Layout

```
/data/
├── qdrant/                  # Qdrant local-mode collections (jobs_dense, jobs_multivec)
├── hf_cache/                # HF Hub model cache (env: HF_HOME=/data/hf_cache)
└── mlruns/                  # MLflow tracking + artifact store (only if running training in-Space)
```

## Storage budget (rough)

| Use | Size | Notes |
|---|---|---|
| `jobs_dense` (240k chunks × 1024-dim int8 + payload) | ~250 MB | HNSW + scalar quantization |
| `jobs_multivec` (ColBERT, ~128-dim/token, int8 PQ) | ~3 GB | Computed once during indexing |
| HF model cache (bge-m3 + reranker + Qwen2.5-7B + NuExtract) | ~16 GB | Qwen dominates; redownloads on first use after restart |
| Headroom | ~1 GB | Drift report HTML, temp parquet, logs |

If we exceed 20 GB after Phase 7, the first thing to evict is the multi-vector
collection (rebuild on demand) or move the Qwen weights to a smaller
fine-tuned variant.

## What does NOT live on persistent storage
- Source parquet snapshots — those live on the HF Dataset repo, fetched on demand.
- Model checkpoints — fetched from HF Hub Model repos as needed.
- Eval reports — committed to the Dataset repo under `reports/`.
