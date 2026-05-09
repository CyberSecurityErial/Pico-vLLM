# Pico-vLLM Test Layout

Tests are organized by the local CI level that runs them:

```text
00_env/                    environment and import checks
01_ops/                    CPU/Torch ops, tiny CPU model, CUDA/Triton ops when available
02_single_card/            single CUDA card smoke tests
03_single_node_multi_card/ single-node multi-GPU communication smoke tests
04_multi_card/             multi-card / tensor-parallel smoke tests
legacy/                    historical scripts not yet migrated into the local CI gate
```

Run the full local gate:

```bash
.venv/bin/python scripts/local_ci.py
```

Run a single level:

```bash
.venv/bin/python scripts/local_ci.py --layer 01_ops
```

The default gate does not require downloaded Qwen weights. Tiny model tests use
randomly initialized small configs so CPU-only environments still exercise the
prefill/decode path.
