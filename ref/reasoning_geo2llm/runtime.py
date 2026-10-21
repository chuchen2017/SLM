from __future__ import annotations

import os
import sys

from reasoning_geo2llm.config import RuntimeConfig

_LAUNCHED_ENV = "GEO2LLM_DISTRIBUTED_LAUNCHED"


def parse_cuda_visible_devices(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def apply_runtime_config(config: RuntimeConfig) -> None:
    devices = parse_cuda_visible_devices(config.cuda_visible_devices)
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def distributed_relaunch_argv(devices: list[str], mixed_precision: str) -> list[str] | None:
    """Argv for a data-parallel relaunch, or None when this process should keep running.

    Listing several GPUs only sets `CUDA_VISIBLE_DEVICES`. A plain `python train.py`
    process still builds one Accelerator and trains on the first visible GPU. Relaunch
    through Accelerate with one process per listed GPU so each device gets the same
    shard of the batch.
    """
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
    # The parent has already imported NumPy, which sets MKL_THREADING_LAYER=INTEL.
    # Accelerate's workers then load libgomp and abort unless this is GNU.
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.execv(sys.executable, argv)
