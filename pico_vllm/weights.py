# weights.py
from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path

import torch
from safetensors import safe_open


_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


@dataclass(frozen=True)
class _WeightMetadata:
    files: list[str]
    key_to_file: dict[str, str]
    total_numel: int
    largest_tensor_numel: int


@dataclass(frozen=True)
class _LoadPlan:
    dtype: torch.dtype
    streaming: bool
    metadata: _WeightMetadata


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{value:.2f}TiB"


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported weight dtype for memory planning: {dtype}")
    return _DTYPE_BYTES[dtype]


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages) * int(page_size)


def _scan_safetensors(weight_dir: str) -> _WeightMetadata:
    safetensor_files = sorted(glob.glob(os.path.join(weight_dir, "*.safetensors")))
    key_to_file: dict[str, str] = {}
    total_numel = 0
    largest_tensor_numel = 0

    for f in safetensor_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for key in st.keys():
                shape = st.get_slice(key).get_shape()
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                key_to_file[key] = f
                total_numel += numel
                largest_tensor_numel = max(largest_tensor_numel, numel)

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {weight_dir!r}")

    return _WeightMetadata(
        files=safetensor_files,
        key_to_file=key_to_file,
        total_numel=total_numel,
        largest_tensor_numel=largest_tensor_numel,
    )


def _floating_module_bytes(model: torch.nn.Module, dtype: torch.dtype | None = None) -> int:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage_id = tensor.untyped_storage().data_ptr()
        if storage_id in seen:
            continue
        seen.add(storage_id)

        if dtype is not None and tensor.is_floating_point():
            total += tensor.numel() * _dtype_nbytes(dtype)
        else:
            total += tensor.numel() * tensor.element_size()
    return total


def _coerce_requested_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return None
    if dtype in ("float32", "fp32", "f32"):
        return torch.float32
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in ("float16", "fp16", "f16"):
        return torch.float16
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unsupported weight dtype: {dtype!r}")


def _select_load_plan(
    model: torch.nn.Module,
    weight_dir: str,
    dtype: torch.dtype | str | None,
    memory_limit_fraction: float,
    verbose: bool,
) -> _LoadPlan:
    metadata = _scan_safetensors(weight_dir)
    requested_dtype = _coerce_requested_dtype(dtype)
    target_dtype = requested_dtype or torch.float32
    streaming = False

    available = _available_memory_bytes()
    if available is None:
        return _LoadPlan(dtype=target_dtype, streaming=streaming, metadata=metadata)

    budget = int(available * memory_limit_fraction)
    bulk_bytes = metadata.total_numel * _dtype_nbytes(target_dtype)

    if requested_dtype is None and target_dtype == torch.float32 and bulk_bytes > budget:
        half_dtype = torch.bfloat16
        half_bytes = metadata.total_numel * _dtype_nbytes(half_dtype)
        if verbose:
            print(
                "内存检测: f32 批量加载预计需要 "
                f"{_format_bytes(bulk_bytes)}，当前预算 {_format_bytes(budget)}，"
                f"切换到 {half_dtype}（权重开销约减半）。",
                flush=True,
            )
        target_dtype = half_dtype

        current_model_bytes = _floating_module_bytes(model)
        half_model_bytes = _floating_module_bytes(model, dtype=half_dtype)
        available_after_cast = available + max(0, current_model_bytes - half_model_bytes)
        half_budget = int(available_after_cast * memory_limit_fraction)
        if half_bytes > half_budget:
            streaming = True
            if verbose:
                print(
                    "内存检测: 半精度批量加载预计仍需要 "
                    f"{_format_bytes(half_bytes)}，预算 {_format_bytes(half_budget)}，"
                    "切换到逐 tensor streaming 加载，避免额外持有完整 state_dict。",
                    flush=True,
                )
    elif bulk_bytes > budget:
        streaming = True
        if verbose:
            print(
                "内存检测: 批量加载预计需要 "
                f"{_format_bytes(bulk_bytes)}，当前预算 {_format_bytes(budget)}，"
                "切换到逐 tensor streaming 加载。",
                flush=True,
            )

    return _LoadPlan(dtype=target_dtype, streaming=streaming, metadata=metadata)


def _read_weight(
    key_to_file: dict[str, str],
    key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    with safe_open(key_to_file[key], framework="pt", device="cpu") as st:
        return st.get_tensor(key).to(dtype=dtype)


def _load_bulk_state_dict(metadata: _WeightMetadata, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state_dict = {}
    for f in metadata.files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for key in st.keys():
                state_dict[key] = st.get_tensor(key).to(dtype=dtype)
    return state_dict


def load_weights(
    model,
    weight_dir: str,
    tp_size: int = 1,
    tp_rank: int = 0,
    dtype: torch.dtype | str | None = None,
    memory_limit_fraction: float = 0.8,
    verbose: bool = True,
):
    """从 safetensors 文件加载权重到 model。

    dtype=None/"auto" 时优先尝试 float32；如果当前内存预算不足，会降级到
    bfloat16。半精度仍不足时切换为逐 tensor streaming 加载，降低峰值内存。
    """

    plan = _select_load_plan(
        model=model,
        weight_dir=weight_dir,
        dtype=dtype,
        memory_limit_fraction=memory_limit_fraction,
        verbose=verbose,
    )

    if plan.dtype != torch.float32:
        current_dtype = next(model.parameters()).dtype
        if current_dtype != plan.dtype:
            if verbose:
                print(
                    f"内存检测: 将模型常驻 dtype 从 {current_dtype} 调整为 {plan.dtype}。",
                    flush=True,
                )
            model = model.to(dtype=plan.dtype)

    if plan.streaming:
        get_tensor = lambda key: _read_weight(plan.metadata.key_to_file, key, plan.dtype)
        tensor_count = len(plan.metadata.key_to_file)
    else:
        state_dict = _load_bulk_state_dict(plan.metadata, plan.dtype)
        get_tensor = state_dict.__getitem__
        tensor_count = len(state_dict)

    # embed_tokens 和 final norm：每卡完整副本，不切
    model.embed_tokens.weight.data = get_tensor("model.embed_tokens.weight")
    model.norm.weight.data = get_tensor("model.norm.weight")

    for i in range(model.cfg.num_hidden_layers):
        layer = model.layers[i]
        p = f"model.layers.{i}"

        # === Attention ===
        # QKV：列切（按 output dim 切，每卡分到一部分 head）
        q_w = get_tensor(f"{p}.self_attn.q_proj.weight")  # (q_size, hidden)
        k_w = get_tensor(f"{p}.self_attn.k_proj.weight")  # (kv_size, hidden)
        v_w = get_tensor(f"{p}.self_attn.v_proj.weight")  # (kv_size, hidden)
        q_b = get_tensor(f"{p}.self_attn.q_proj.bias")
        k_b = get_tensor(f"{p}.self_attn.k_proj.bias")
        v_b = get_tensor(f"{p}.self_attn.v_proj.bias")

        if tp_size > 1:
            q_w = q_w.chunk(tp_size, dim=0)[tp_rank]
            k_w = k_w.chunk(tp_size, dim=0)[tp_rank]
            v_w = v_w.chunk(tp_size, dim=0)[tp_rank]
            q_b = q_b.chunk(tp_size, dim=0)[tp_rank]
            k_b = k_b.chunk(tp_size, dim=0)[tp_rank]
            v_b = v_b.chunk(tp_size, dim=0)[tp_rank]

        layer.attn.qkv_proj.weight.data = torch.cat([q_w, k_w, v_w], dim=0)
        layer.attn.qkv_proj.bias.data = torch.cat([q_b, k_b, v_b], dim=0)

        # O proj：行切（按 input dim 切，每卡拿到部分输入，forward 后 All-Reduce）
        o_w = get_tensor(f"{p}.self_attn.o_proj.weight")  # (hidden, q_size)
        if tp_size > 1:
            o_w = o_w.chunk(tp_size, dim=1)[tp_rank]  # 注意是 dim=1（input dim）
        layer.attn.o_proj.weight.data = o_w

        # === FFN ===
        # gate_up：列切
        gate_w = get_tensor(f"{p}.mlp.gate_proj.weight")  # (intermediate, hidden)
        up_w = get_tensor(f"{p}.mlp.up_proj.weight")      # (intermediate, hidden)
        if tp_size > 1:
            gate_w = gate_w.chunk(tp_size, dim=0)[tp_rank]
            up_w = up_w.chunk(tp_size, dim=0)[tp_rank]
        layer.ffn.gate_up_proj.weight.data = torch.cat([gate_w, up_w], dim=0)

        # down：行切
        down_w = get_tensor(f"{p}.mlp.down_proj.weight")  # (hidden, intermediate)
        if tp_size > 1:
            down_w = down_w.chunk(tp_size, dim=1)[tp_rank]  # dim=1（input dim）
        layer.ffn.down_proj.weight.data = down_w

        # RMSNorm：每卡完整副本，不切
        layer.norm1.weight.data = get_tensor(f"{p}.input_layernorm.weight")
        layer.norm2.weight.data = get_tensor(f"{p}.post_attention_layernorm.weight")

    if tp_rank == 0:
        print(f"权重加载完成，共 {tensor_count} 个 tensor, tp_size={tp_size}")
    return model

# 在 weights.py 末尾测试
if __name__ == "__main__":
    from model import Qwen25_15B, ModelConfig
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    print("加载成功")
