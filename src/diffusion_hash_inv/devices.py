"""Select an available PyTorch backend and time completed accelerator work."""

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N") from error
    if device.type == "mps":
        if device.index not in (None, 0):
            raise ValueError("MPS supports only device 0")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable to this process. Use a Metal-capable host terminal; "
                "a sandbox can hide the GPU even when PyTorch has MPS support. "
                "Use --device cpu explicitly for CPU execution."
            )
        return torch.device("mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use an available device explicitly")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        return torch.device("cuda", index)
    if device.type != "cpu" or device.index not in (None, 0):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
