"""AMP (autocast) helpers shared between training and evaluation."""

from __future__ import annotations


def amp_dtype_from_name(name: str):
    """Map a YAML-friendly name to a torch dtype."""
    import torch

    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "bf16": torch.bfloat16}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown AMP dtype '{name}'. Supported: {sorted(table.keys())}."
        ) from exc


def pick_amp_dtype(name: str, device):
    """Choose ``(use_amp, dtype)`` for ``--amp auto`` / off / explicit names.

    On Ampere+ GPUs we prefer bf16; on Turing we fall back to fp16. Pure CPU
    runs always disable AMP.
    """
    import torch

    if name == "off" or getattr(device, "type", str(device)) == "cpu":
        return False, torch.float32
    if name == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return (True, torch.bfloat16) if major >= 8 else (True, torch.float16)
        return False, torch.float32
    return True, amp_dtype_from_name(name)
