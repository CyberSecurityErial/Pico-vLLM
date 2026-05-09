import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


PACKAGE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_DIR))

from blockmanager import BlockManager
from model import ModelConfig, Qwen25_15B


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert world_size >= 2, "tiny TP smoke requires at least 2 ranks"
    assert torch.cuda.is_available(), "CUDA is required"

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        use_cuda=True,
        tp_size=world_size,
        tp_rank=rank,
        tp_group=dist.group.WORLD,
    )
    dtype = torch.float32
    model = Qwen25_15B(cfg).to(device=device, dtype=dtype).eval()
    bm = BlockManager(
        num_gpu_blocks=8,
        num_cpu_blocks=0,
        block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=dtype,
        device=device,
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    seq_len = input_ids.shape[1]
    block_table = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    block_table[0, 0] = 0

    with torch.no_grad():
        logits = model(
            input_ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0),
            slot_mapping=torch.arange(seq_len, dtype=torch.int32, device=device),
            is_prefill=True,
            block_table=block_table,
            context_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            new_token_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            q_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
        )

    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
