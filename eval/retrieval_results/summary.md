# Retrieval eval — multi-variant comparison

| Variant | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 | latency |
|---|---|---|---|---|---|---|---|
| `dense` | 48 | 0.291 | 0.363 | 0.393 | 0.412 | 0.349 | 186 ms/q |
| `hybrid+rerank` | 48 | 0.421 | 0.486 | 0.511 | 0.518 | 0.476 | 700 ms/q |
