from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TestEnvironment:
    python: str
    torch_version: str
    cuda_available: bool
    gpu_count: int
    triton_available: bool
    pytest_available: bool
    transformers_available: bool
    safetensors_available: bool
    weights_available: bool
    torchrun_available: bool


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def weights_available(repo_root: Path = REPO_ROOT) -> bool:
    weights_dir = repo_root / "weights"
    required = ("model.safetensors", "tokenizer.json")
    return weights_dir.is_dir() and all((weights_dir / name).exists() for name in required)


def detect_environment(repo_root: Path = REPO_ROOT) -> TestEnvironment:
    cuda_available = torch.cuda.is_available()
    return TestEnvironment(
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        gpu_count=torch.cuda.device_count() if cuda_available else 0,
        triton_available=module_available("triton"),
        pytest_available=module_available("pytest"),
        transformers_available=module_available("transformers"),
        safetensors_available=module_available("safetensors"),
        weights_available=weights_available(repo_root),
        torchrun_available=shutil.which("torchrun") is not None,
    )
