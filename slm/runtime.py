"""Device selection. Several listed GPUs relaunch one process per GPU."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from slm.config import RuntimeConfig

_LAUNCHED_ENV = "SLM_DISTRIBUTED_LAUNCHED"


def prefer_env_libstdcxx() -> None:
    """Load this environment's libstdc++ before NumPy, PyTorch, or SciPy.

    A ``LD_LIBRARY_PATH`` that lists system directories first makes SciPy (pulled
    in by ``transformers``) resolve ``CXXABI`` symbols against an older system
    ``libstdc++``. Loading the env copy first lets those extensions start.
    """
    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    ctypes.CDLL(str(lib), mode=mode)


def parse_cuda_visible_devices(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def apply_runtime_config(config: RuntimeConfig) -> None:
    devices = parse_cuda_visible_devices(config.cuda_visible_devices)
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def distributed_relaunch_argv(devices: list[str], mixed_precision: str) -> list[str] | None:
    if len(devices) <= 1:
        return None
    if os.environ.get(_LAUNCHED_ENV) == "1" or os.environ.get("LOCAL_RANK") is not None:
        return None
    precision = mixed_precision if mixed_precision in {"no", "fp16", "bf16", "fp8"} else "no"
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_machines",
        "1",
        "--num_processes",
        str(len(devices)),
        "--gpu_ids",
        ",".join(devices),
        "--mixed_precision",
        precision,
        "--dynamo_backend",
        "no",
        sys.argv[0],
        *sys.argv[1:],
    ]


def relaunch_distributed_training(devices: list[str], mixed_precision: str) -> None:
    argv = distributed_relaunch_argv(devices, mixed_precision)
    if argv is None:
        return
    os.environ[_LAUNCHED_ENV] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.execv(sys.executable, argv)
