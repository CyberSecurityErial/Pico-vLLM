# Pico-vLLM Test Layout

Tests are organized by the local CI level that runs them:

```text
00_env/                    environment and import checks
01_ops/                    operator-level tests for CPU/Torch and CUDA/Triton
02_single_card/            single-device model inference tests (CPU and CUDA)
03_single_node_multi_card/ single-node multi-GPU communication smoke tests
04_multi_card/             multi-card / tensor-parallel smoke tests
legacy/                    historical scripts not yet migrated into the local CI gate
```

Run the full local gate:

```bash
.venv/bin/python scripts/local_ci.py
```

Each run writes per-step logs under `logs/local_ci/<timestamp>/`, including
stdout/stderr, pytest prints, and a `summary.log`.

Run a single level:

```bash
.venv/bin/python scripts/local_ci.py --layer 01_ops
```

The default gate does not require downloaded Qwen weights. Tiny model tests use
randomly initialized small configs so CPU-only environments still exercise the
prefill/decode path.

Run the single-device model layer, including the real CPU Qwen smoke test when
local `./weights` is available:

```bash
.venv/bin/python scripts/local_ci.py --layer 02_single_card
```

Useful knobs:

```bash
PICO_VLLM_CPU_REAL_PROMPT="Hello" \
PICO_VLLM_CPU_REAL_MAX_NEW_TOKENS=32 \
.venv/bin/python scripts/local_ci.py --layer 02_single_card
```

The test decodes greedily until `tokenizer.eos_token_id` or the safety cap in
`PICO_VLLM_CPU_REAL_MAX_NEW_TOKENS`, and prints load/prefill/decode timing plus
throughput, generated token ids, and generated text.
