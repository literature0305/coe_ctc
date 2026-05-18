"""Model factory.

Picks an encoder architecture and a size preset, wires up the CTC head, and
returns a ready-to-train ``CtcModel``.

CLI:
    python -m coe_ctc.models.builder --arch conformer --size small --vocab-size 500

Size presets target the following parameter counts (encoder + head, vocab=500):
    tiny    ~30M
    small   ~120M
    medium  ~300M
    large   ~800M

These are starting points — actual counts depend on vocab size and head
projection. The factory prints the realized count when invoked from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn

from coe_ctc.data.transforms import SpecAugmentConfig
from coe_ctc.models.coe import CoeModel
from coe_ctc.models.conformer import ConformerEncoder
from coe_ctc.models.ctc import CtcModel
from coe_ctc.models.transformer import TransformerEncoder
from coe_ctc.models.zipformer import ZipformerEncoder


# ─────────────────────────────────────────────────────────────────────────
# Size presets
# ─────────────────────────────────────────────────────────────────────────


_TRANSFORMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny":   {"d_model": 256,  "num_layers": 12, "num_heads": 4,  "d_ff": 1024},
    "small":  {"d_model": 512,  "num_layers": 12, "num_heads": 8,  "d_ff": 2048},
    "medium": {"d_model": 768,  "num_layers": 16, "num_heads": 12, "d_ff": 3072},
    "large":  {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
}

_CONFORMER_PRESETS: Dict[str, Dict[str, Any]] = {
    # Roughly mirrors IceFall / NeMo Conformer-CTC config sizes.
    "tiny":   {"d_model": 256, "num_layers": 12, "num_heads": 4,  "d_ff": 1024, "kernel_size": 31},
    "small":  {"d_model": 384, "num_layers": 16, "num_heads": 6,  "d_ff": 1536, "kernel_size": 31},
    "medium": {"d_model": 512, "num_layers": 18, "num_heads": 8,  "d_ff": 2048, "kernel_size": 31},
    "large":  {"d_model": 768, "num_layers": 24, "num_heads": 12, "d_ff": 3072, "kernel_size": 31},
}

_ZIPFORMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny":   {"d_model": 256, "num_heads": 4,  "d_ff": 1024,
                "downsampling_factors": (1, 2, 4, 2, 1),
                "num_layers_per_stack":  (2, 3, 4, 3, 2)},
    "small":  {"d_model": 384, "num_heads": 6,  "d_ff": 1536,
                "downsampling_factors": (1, 2, 4, 2, 1),
                "num_layers_per_stack":  (2, 4, 6, 4, 2)},
    "medium": {"d_model": 512, "num_heads": 8,  "d_ff": 2048,
                "downsampling_factors": (1, 2, 4, 8, 4, 2, 1),
                "num_layers_per_stack":  (2, 3, 4, 6, 4, 3, 2)},
    "large":  {"d_model": 768, "num_heads": 12, "d_ff": 3072,
                "downsampling_factors": (1, 2, 4, 8, 4, 2, 1),
                "num_layers_per_stack":  (3, 4, 5, 8, 5, 4, 3)},
}


_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "transformer": _TRANSFORMER_PRESETS,
    "conformer":   _CONFORMER_PRESETS,
    "zipformer":   _ZIPFORMER_PRESETS,
}


# ─────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────


def list_archs() -> list[str]:
    return list(_PRESETS.keys())


def list_sizes(arch: str) -> list[str]:
    return list(_PRESETS[arch].keys())


def build_encoder(
    arch: str,
    size: str | None = None,
    *,
    num_features: int = 80,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    **overrides: Any,
) -> nn.Module:
    """Build the bare encoder (no CTC head)."""
    arch = arch.lower()
    if arch not in _PRESETS:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list_archs()}.")

    if size is None:
        base: Dict[str, Any] = {}
    else:
        if size not in _PRESETS[arch]:
            raise ValueError(f"Unknown size '{size}' for arch '{arch}'. Available: {list_sizes(arch)}.")
        base = dict(_PRESETS[arch][size])

    base.update(overrides)
    base.setdefault("num_features", num_features)
    base.setdefault("dropout", dropout)
    base.setdefault("attn_dropout", attn_dropout)

    if arch == "transformer":
        # Transformer presets don't carry kernel_size; drop it if a user passed one via overrides.
        base.pop("kernel_size", None)
        return TransformerEncoder(**base)
    if arch == "conformer":
        return ConformerEncoder(**base)
    if arch == "zipformer":
        return ZipformerEncoder(**base)
    raise RuntimeError(f"Unhandled arch '{arch}'.")


def build_model(
    arch: str,
    size: str | None = None,
    *,
    vocab_size: int,
    num_features: int = 80,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    head_dropout: float = 0.0,
    blank_idx: int = 0,
    **encoder_overrides: Any,
) -> CtcModel:
    """Build a full CtcModel = encoder + CTC head."""
    encoder = build_encoder(
        arch,
        size,
        num_features=num_features,
        dropout=dropout,
        attn_dropout=attn_dropout,
        **encoder_overrides,
    )
    return CtcModel(encoder, vocab_size=vocab_size, blank_idx=blank_idx, head_dropout=head_dropout)


def build_coe_model(
    arch: str,
    size: str | None = None,
    *,
    vocab_size: int,
    num_features: int = 80,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    head_dropout: float = 0.0,
    blank_idx: int = 0,
    coe_cfg: dict,
    spec_aug_cfg: SpecAugmentConfig | None = None,
    **encoder_overrides: Any,
) -> CoeModel:
    """Build a Chain-of-Encoders CTC model from the YAML ``coe:`` section."""
    encoder = build_encoder(
        arch,
        size,
        num_features=num_features,
        dropout=dropout,
        attn_dropout=attn_dropout,
        **encoder_overrides,
    )
    num_chains = int(coe_cfg.get("num_enc_chains", 5))
    ratios = coe_cfg.get("time_mask_ratios") or [0.1] * num_chains
    return CoeModel(
        encoder,
        vocab_size=vocab_size,
        blank_idx=blank_idx,
        head_dropout=head_dropout,
        num_enc_chains=num_chains,
        time_mask_ratios=ratios,
        loss_alphas=coe_cfg.get("loss_alphas"),
        pre_enc_layer_idx=coe_cfg.get("pre_enc_layer_idx", -1),
        pre_enc_feature_detach=bool(coe_cfg.get("pre_enc_feature_detach", False)),
        head_share=bool(coe_cfg.get("head_share", False)),
        spec_aug_cfg=spec_aug_cfg,
    )


# ─────────────────────────────────────────────────────────────────────────
# Standalone CLI — quick smoke test / param-count introspection
# ─────────────────────────────────────────────────────────────────────────


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build a CtcModel and print its parameter count.")
    p.add_argument("--arch", required=True, choices=list_archs())
    p.add_argument("--size", default=None, choices=["tiny", "small", "medium", "large"])
    p.add_argument("--vocab-size", type=int, default=500)
    p.add_argument("--num-features", type=int, default=80)
    p.add_argument(
        "--shape-test",
        action="store_true",
        help="Run a tiny forward pass with a random batch to verify shapes.",
    )
    args = p.parse_args()

    model = build_model(
        args.arch,
        args.size,
        vocab_size=args.vocab_size,
        num_features=args.num_features,
    )
    total = model.num_parameters()
    by_part = model.parameter_breakdown()
    print(f"arch={args.arch} size={args.size or '-'} vocab={args.vocab_size}")
    print(f"Total trainable params: {total:,}  (~{total/1e6:.1f}M)")
    for k, v in by_part.items():
        print(f"  {k:>10s}: {v:>14,}  ({100*v/total:5.1f}%)")

    if args.shape_test:
        import torch

        b, t = 2, 320
        x = torch.randn(b, t, args.num_features)
        x_lens = torch.tensor([t, t - 20], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            out = model(x, x_lens)
        print(f"forward OK: encoded={tuple(out.encoded.shape)} lens={out.encoded_lens.tolist()} "
              f"log_probs={tuple(out.log_probs.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
