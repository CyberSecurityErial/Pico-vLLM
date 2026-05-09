import os
import sys
from pathlib import Path

import pytest
import torch


TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PACKAGE_DIR))

def pytest_ignore_collect(collection_path, config):
    if os.environ.get("PICO_VLLM_COLLECT_LEGACY_TESTS") == "1":
        return False

    path = Path(str(collection_path))
    return path.is_relative_to(TESTS_DIR) and "legacy" in path.relative_to(TESTS_DIR).parts


def _weights_available() -> bool:
    weights_dir = REPO_ROOT / "weights"
    required = ("model.safetensors", "tokenizer.json")
    return weights_dir.is_dir() and all((weights_dir / name).exists() for name in required)


def pytest_collection_modifyitems(config, items):
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    has_weights = _weights_available()

    skip_cuda = pytest.mark.skip(reason="CUDA is not available")
    skip_weights = pytest.mark.skip(reason="./weights is not available")

    for item in items:
        if "cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)

        if "weights" in item.keywords and not has_weights:
            item.add_marker(skip_weights)

        min_gpus = item.get_closest_marker("min_gpus")
        if min_gpus is not None:
            required = int(min_gpus.args[0])
            if gpu_count < required:
                item.add_marker(
                    pytest.mark.skip(reason=f"requires at least {required} CUDA devices")
                )
