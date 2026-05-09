#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pico_vllm" / "tests"))

from ci_utils import TestEnvironment, detect_environment


Requirement = Callable[[TestEnvironment], tuple[bool, str]]


@dataclass(frozen=True)
class Step:
    layer: str
    name: str
    command: list[str]
    requirement: Requirement


def always(_: TestEnvironment) -> tuple[bool, str]:
    return True, ""


def requires_cuda(env: TestEnvironment) -> tuple[bool, str]:
    if env.cuda_available:
        return True, ""
    return False, "CUDA is not available"


def requires_two_gpus(env: TestEnvironment) -> tuple[bool, str]:
    if env.gpu_count >= 2:
        return True, ""
    return False, f"requires at least 2 CUDA devices, found {env.gpu_count}"


def pytest_cmd(*paths: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", *paths]


def torchrun_cmd(nproc: int, script: str, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        script,
        *args,
    ]


def build_steps() -> list[Step]:
    return [
        Step(
            layer="00_env",
            name="syntax compile",
            command=[sys.executable, "-m", "compileall", "-q", "pico_vllm", "scripts"],
            requirement=always,
        ),
        Step(
            layer="00_env",
            name="environment detection",
            command=pytest_cmd("pico_vllm/tests/00_env/test_environment.py"),
            requirement=always,
        ),
        Step(
            layer="01_ops",
            name="CPU torch operators",
            command=pytest_cmd("pico_vllm/tests/01_ops/test_torch_cpu_ops.py"),
            requirement=always,
        ),
        Step(
            layer="01_ops",
            name="CPU tiny random model inference",
            command=pytest_cmd("pico_vllm/tests/01_ops/test_torch_cpu_tiny_model.py"),
            requirement=always,
        ),
        Step(
            layer="01_ops",
            name="CUDA paged prefill attention",
            command=pytest_cmd("pico_vllm/tests/01_ops/test_triton_prefill_attention.py"),
            requirement=requires_cuda,
        ),
        Step(
            layer="02_single_card",
            name="single CUDA card tiny model",
            command=pytest_cmd("pico_vllm/tests/02_single_card/test_tiny_model_smoke.py"),
            requirement=requires_cuda,
        ),
        Step(
            layer="03_single_node_multi_card",
            name="single-node NCCL all-reduce smoke",
            command=torchrun_cmd(
                2,
                "pico_vllm/tests/03_single_node_multi_card/test_nccl_smoke.py",
            ),
            requirement=requires_two_gpus,
        ),
        Step(
            layer="04_multi_card",
            name="tiny tensor-parallel model smoke",
            command=torchrun_cmd(2, "pico_vllm/tests/04_multi_card/test_tiny_tp_smoke.py"),
            requirement=requires_two_gpus,
        ),
    ]


def print_environment(env: TestEnvironment) -> None:
    print("Detected test environment:", flush=True)
    print(f"  python: {env.python}", flush=True)
    print(f"  torch: {env.torch_version}", flush=True)
    print(f"  cuda_available: {env.cuda_available}", flush=True)
    print(f"  gpu_count: {env.gpu_count}", flush=True)
    print(f"  triton_available: {env.triton_available}", flush=True)
    print(f"  pytest_available: {env.pytest_available}", flush=True)
    print(f"  transformers_available: {env.transformers_available}", flush=True)
    print(f"  safetensors_available: {env.safetensors_available}", flush=True)
    print(f"  weights_available: {env.weights_available}", flush=True)
    print(f"  torchrun_available: {env.torchrun_available}", flush=True)
    print(flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layered Pico-vLLM local CI gate.")
    parser.add_argument(
        "--layer",
        action="append",
        choices=[
            "00_env",
            "01_ops",
            "02_single_card",
            "03_single_node_multi_card",
            "04_multi_card",
        ],
        help="Run only the selected layer. May be passed multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected steps and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = detect_environment(REPO_ROOT)
    print_environment(env)

    selected_layers = set(args.layer or [])
    steps = build_steps()
    if selected_layers:
        steps = [step for step in steps if step.layer in selected_layers]

    if args.list:
        for step in steps:
            ok, reason = step.requirement(env)
            status = "run" if ok else f"skip: {reason}"
            print(f"[{step.layer}] {step.name}: {status}", flush=True)
        return 0

    failed = False
    for step in steps:
        ok, reason = step.requirement(env)
        label = f"[{step.layer}] {step.name}"
        if not ok:
            print(f"SKIP {label}: {reason}", flush=True)
            continue

        print(f"RUN  {label}", flush=True)
        print("     " + " ".join(step.command), flush=True)
        result = subprocess.run(step.command, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"FAIL {label}: exit code {result.returncode}", flush=True)
            failed = True
            break
        print(f"PASS {label}\n", flush=True)

    if failed:
        return 1

    print("Local CI completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
