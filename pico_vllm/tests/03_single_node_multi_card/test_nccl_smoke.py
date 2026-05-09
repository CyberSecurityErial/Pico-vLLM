import os

import torch
import torch.distributed as dist


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert world_size >= 2, "single-node multi-GPU smoke requires at least 2 ranks"
    assert torch.cuda.is_available(), "CUDA is required"

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    assert value.item() == expected

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
