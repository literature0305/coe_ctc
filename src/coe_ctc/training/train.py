"""Training entry point.

Run via:
    python -m coe_ctc.training.train --data libri100 --config <path/to/yaml> [--set k=v ...]

Features (all toggled from YAML):
    • Single-GPU or DDP via torchrun (auto-detected from WORLD_SIZE / LOCAL_RANK).
    • Mixed-precision (bf16 default on A100/H100, fp32 on Turing).
    • Gradient accumulation.
    • Gradient clipping (max_grad_norm).
    • Checkpointing (latest + top-K best by validation WER).
    • Resume from checkpoint (model + optimizer + scheduler + RNG).
    • TensorBoard scalars / images.
    • TRAINING HEALTH REPORT every `health.interval_steps` (errlog019-2 style).
    • Validation every epoch on dev-other → BENCHMARK EVALUATION + REF/HYP samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────
# Path bootstrap so the file works when invoked directly OR as a module.
# ─────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from coe_ctc.data.bpe import default_bpe_path
from coe_ctc.data.datamodule import CtcDataModule, DataModuleConfig
from coe_ctc.data.librispeech import DATASETS, manifest_prefix_for
from coe_ctc.data.transforms import SpecAugment, SpecAugmentConfig, PerUtteranceMVN
from coe_ctc.models.builder import build_coe_model, build_model
from coe_ctc.training.checkpoint import (
    TopKCheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from coe_ctc.training.health import (
    HealthReporterConfig,
    TrainingHealthReporter,
)
from coe_ctc.training.optimizer import OptimConfig, build_optimizer, build_scheduler
from coe_ctc.training.validation import Validator
from coe_ctc.training.visualize import render_validation_plots
from coe_ctc.utils.amp import amp_dtype_from_name
from coe_ctc.utils.config import load_yaml, merge_overrides, resolve_paths
from coe_ctc.utils.distributed import (
    is_distributed,
    is_main_process,
    log_distributed_info,
    main_process_first,
    setup_distributed,
    setup_seeds,
    teardown_distributed,
    unwrap_model,
)
from coe_ctc.utils.logging import setup_logger


# ─────────────────────────────────────────────────────────────────────────
# CLI parsing
# ─────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="coe_ctc.training.train", description="CTC ASR training.")
    p.add_argument("--data", required=True, help="Dataset alias (libri100 / libri960 / libri_light).")
    p.add_argument("--config", required=True, help="Training YAML.")
    p.add_argument("--set", action="append", default=[], help="Inline override key=value (repeatable).")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps from config.")
    p.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers.")
    p.add_argument(
        "--n_bpe",
        type=int,
        default=None,
        help=(
            "BPE vocab size shortcut — resolves data.bpe_model to "
            "downloads/bpe/<data>_bpe<N>/spm.model. Overrides the value baked "
            "into the YAML config and any --set data.bpe_model=… argument."
        ),
    )
    p.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint.pt to resume from.")
    p.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    p.add_argument("--dry-run", action="store_true", help="Build everything, then exit before train loop.")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────
# Helper to summarize state for the config banner
# ─────────────────────────────────────────────────────────────────────────


def _human_int(n: int) -> str:
    return f"{n:,}"


def _summarize_model(model: nn.Module, arch: str, size: str, vocab_size: int) -> dict:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {}
    for name, mod in model.named_children():
        parts[name] = sum(p.numel() for p in mod.parameters())
    s = {
        "Architecture": f"{arch} / {size}",
        "Parameters": f"{_human_int(total)} (~{total/1e6:.1f}M)",
        "Vocab size": f"{vocab_size}",
    }
    for k, v in parts.items():
        s[f"  {k}"] = f"{_human_int(v)}  ({100*v/total:.1f}%)"
    return s


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.data not in DATASETS:
        raise SystemExit(
            f"Unknown --data alias '{args.data}'. Available: {sorted(DATASETS)}."
        )

    # ----- Config -----
    cfg = load_yaml(args.config)
    cfg = merge_overrides(cfg, args.set)
    if args.max_steps is not None:
        cfg.setdefault("optim", {})["max_steps"] = args.max_steps
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers
    if args.n_bpe is not None:
        cfg.setdefault("data", {})["bpe_model"] = str(_REPO_ROOT / default_bpe_path(args.data, args.n_bpe))
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg = resolve_paths(cfg, _REPO_ROOT)

    # ----- DDP -----
    rank, local_rank, world_size = setup_distributed(backend=cfg.get("distributed_backend", "nccl"))
    seed = int(cfg.get("seed", 42))
    setup_seeds(seed)

    # ----- Output dir -----
    output_dir = Path(cfg.get("output_dir") or f"outputs/{Path(args.config).stem}")
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "best_checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
        # Freeze the config so resume reads the same hparams.
        (output_dir / "config.yaml").write_text(json.dumps(cfg, indent=2))

    # ----- Logger -----
    log_file = str(output_dir / "train.log") if is_main_process() else None
    setup_logger("coe_ctc", log_file=log_file, rank=rank)
    logger = logging.getLogger("coe_ctc.train")
    log_distributed_info(logger)
    logger.info(f"Config file : {args.config}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Data alias  : {args.data}")
    logger.info(f"Seed        : {seed}")

    # ----- Build datamodule -----
    dcfg_dict = dict(cfg.get("data", {}))
    dcfg_dict.setdefault("manifest_dir", str(_REPO_ROOT / "downloads" / "manifests" / args.data))
    dcfg_dict.setdefault("feats_dir", str(_REPO_ROOT / "downloads" / "feats" / f"{args.data}_fbank80"))
    dcfg_dict.setdefault("manifest_prefix", manifest_prefix_for(args.data))
    if "bpe_model" not in dcfg_dict or not dcfg_dict["bpe_model"]:
        # Default to the largest standard vocab (3000) — matches the size in
        # the spec command `--n_bpe [3000, 300]`.
        dcfg_dict["bpe_model"] = str(_REPO_ROOT / default_bpe_path(args.data, 3000))
    dm_cfg = DataModuleConfig(**{k: v for k, v in dcfg_dict.items() if hasattr(DataModuleConfig, k) or k in DataModuleConfig.__dataclass_fields__})
    logger.info(f"Manifest dir: {dm_cfg.manifest_dir}")
    logger.info(f"BPE model   : {dm_cfg.bpe_model}")

    with main_process_first():
        dm = CtcDataModule(dm_cfg)
        train_loader = dm.train_loader()
        valid_loader = dm.valid_loader()
    assert dm.tokenizer is not None
    vocab_size = dm.tokenizer.vocab_size
    logger.info(f"BPE vocab   : {vocab_size}")

    # ----- Build model -----
    mcfg = cfg.get("model", {})
    arch = mcfg.get("arch", "conformer")
    size = mcfg.get("size", "small")
    encoder_overrides: dict[str, Any] = {k: v for k, v in mcfg.items() if k not in ("arch", "size", "head_dropout", "blank_idx")}

    coe_cfg = cfg.get("coe")
    is_coe = bool(coe_cfg)
    if is_coe and dm_cfg.apply_spec_augment:
        # CoE owns freq + per-pass time masking internally; data-side
        # SpecAugment would double-mask.
        logger.warning(
            "CoE mode: forcing data.apply_spec_augment=false (model-side "
            "masking handles freq + per-pass time masks)."
        )
        dm_cfg.apply_spec_augment = False

    sa_cfg = SpecAugmentConfig(
        num_freq_masks=dm_cfg.spec_aug_num_freq_masks,
        freq_mask_param=dm_cfg.spec_aug_freq_mask_param,
        num_time_masks=dm_cfg.spec_aug_num_time_masks,
        time_mask_param=dm_cfg.spec_aug_time_mask_param,
        time_mask_ratio=dm_cfg.spec_aug_time_mask_ratio,
        apply_prob=1.0 if dm_cfg.apply_spec_augment else 0.0,
    )

    if is_coe:
        model = build_coe_model(
            arch=arch,
            size=size,
            vocab_size=vocab_size,
            num_features=int(cfg.get("data", {}).get("num_features", 80)),
            head_dropout=float(mcfg.get("head_dropout", 0.0)),
            blank_idx=int(mcfg.get("blank_idx", 0)),
            coe_cfg=coe_cfg,
            spec_aug_cfg=sa_cfg,
            **encoder_overrides,
        )
    else:
        model = build_model(
            arch=arch,
            size=size,
            vocab_size=vocab_size,
            num_features=int(cfg.get("data", {}).get("num_features", 80)),
            head_dropout=float(mcfg.get("head_dropout", 0.0)),
            blank_idx=int(mcfg.get("blank_idx", 0)),
            **encoder_overrides,
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
    raw_model = unwrap_model(model)

    spec_aug = None if is_coe else SpecAugment(sa_cfg).to(device)
    mvn = PerUtteranceMVN().to(device) if dm_cfg.per_utt_mvn else None

    # ----- Optimizer + scheduler -----
    ocfg = OptimConfig(**{k: v for k, v in cfg.get("optim", {}).items() if k in OptimConfig.__dataclass_fields__})
    optimizer = build_optimizer(model, ocfg)
    scheduler = build_scheduler(optimizer, ocfg)
    grad_accum_steps = max(1, int(cfg.get("optim", {}).get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg.get("optim", {}).get("max_grad_norm", 5.0))
    max_steps = int(ocfg.max_steps)

    # ----- AMP -----
    amp_cfg = cfg.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", True))
    amp_dtype_name = str(amp_cfg.get("dtype", "bfloat16"))
    amp_dtype = amp_dtype_from_name(amp_dtype_name)
    # GradScaler needed only for float16; bf16 doesn't need scaling.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # ----- Health reporter -----
    hcfg = HealthReporterConfig(**{k: v for k, v in cfg.get("health", {}).items() if k in HealthReporterConfig.__dataclass_fields__})
    health = TrainingHealthReporter(hcfg, max_steps=max_steps)
    health.grad_accum_steps = grad_accum_steps
    if is_coe:
        health.coe_passes = raw_model.num_enc_chains

    # ----- Validator -----
    val_cfg = cfg.get("validation", {})
    validator = Validator(
        tokenizer=dm.tokenizer,
        blank_idx=int(mcfg.get("blank_idx", 0)),
        sample_count=int(val_cfg.get("num_samples", 10)),
        output_dir=output_dir,
        device=device,
    )

    # ----- Checkpoint manager -----
    ckpt_cfg = cfg.get("checkpoint", {})
    top_k_mgr = TopKCheckpointManager(output_dir / "best_checkpoints", top_k=int(ckpt_cfg.get("top_k", 5)))
    rolling_keep = int(ckpt_cfg.get("rolling_keep", 3))
    save_every_steps = int(ckpt_cfg.get("save_every_steps", 2000))
    validate_every_epochs = int(val_cfg.get("every_epochs", 1))

    # ----- TensorBoard -----
    tb_writer = None
    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        except ImportError:
            logger.warning("TensorBoard not available; skipping scalar logging.")

    # ----- Resume -----
    step = 0
    epoch = 0
    if args.resume_from_checkpoint:
        loaded = load_checkpoint(
            args.resume_from_checkpoint,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            map_location=device,
            strict=True,
            restore_rng=True,
        )
        step = int(loaded.get("step", 0))
        epoch = int(loaded.get("epoch", 0))
        validator.best_wer = float(loaded.get("best_wer", float("inf")))
        logger.info(f"Resumed from {args.resume_from_checkpoint} at step={step} epoch={epoch}")

    # ----- Config banner (rank-0) -----
    if is_main_process():
        model_summary = _summarize_model(raw_model, arch, size, vocab_size)
        if is_coe:
            model_summary["CoE chains"] = str(raw_model.num_enc_chains)
            model_summary["CoE alphas"] = ", ".join(f"{a:.3f}" for a in raw_model.alphas)
            model_summary["CoE r_m"] = ", ".join(f"{r:.2f}" for r in raw_model.masking.time_mask_ratios)
            model_summary["CoE pre-layer"] = str(raw_model.pre_enc_layer_idx)
            model_summary["CoE detach"] = "yes" if raw_model.pre_enc_feature_detach else "no"
            model_summary["CoE head_share"] = "yes" if raw_model.head_share else "no"
        sa_label = "on" if dm_cfg.apply_spec_augment else ("CoE-internal" if is_coe else "off")
        health.log_config(
            model_summary=model_summary,
            data_summary={
                "Dataset": args.data,
                "Train parts": ", ".join(dm_cfg.train_parts),
                "Dev part": dm_cfg.dev_part,
                "Test parts": ", ".join(dm_cfg.test_parts),
                "Manifest dir": dm_cfg.manifest_dir,
                "Features dim": str(int(cfg.get("data", {}).get("num_features", 80))),
                "BPE model": dm_cfg.bpe_model,
                "Bucket max_dur": f"{dm_cfg.max_duration}s",
                "Num buckets": str(dm_cfg.num_buckets),
                "SpecAugment": sa_label,
            },
            optim_summary={
                "Optimizer": ocfg.name,
                "Peak LR": f"{ocfg.lr:.2e}",
                "Scheduler": ocfg.scheduler,
                "Warmup steps": str(ocfg.warmup_steps),
                "Max steps": str(ocfg.max_steps),
                "Weight decay": str(ocfg.weight_decay),
                "Max grad norm": str(max_grad_norm),
                "Grad accum": str(grad_accum_steps),
                "AMP": f"{'on' if use_amp else 'off'} ({amp_dtype_name})",
            },
            distributed_summary={
                "World size": str(world_size),
                "Backend": cfg.get("distributed_backend", "nccl"),
                "DDP": "yes" if is_distributed() else "no",
            },
            monitoring_summary={
                "Health steps": str(hcfg.interval_steps),
                "Save steps": str(save_every_steps),
                "Top-K ckpts": str(top_k_mgr.top_k),
                "Valid every": f"{validate_every_epochs} epoch(s)",
                "Output dir": str(output_dir),
                "TensorBoard": "on" if tb_writer is not None else "off",
            },
        )

    if args.dry_run:
        logger.info("[dry-run] exiting before training loop.")
        teardown_distributed()
        return 0

    # ----- Training loop -----
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0
    accum_batch_utt = 0

    while step < max_steps:
        epoch += 1
        if is_main_process():
            logger.info(f"==================== Epoch {epoch} start ====================")
        # Lhotse DynamicBucketingSampler is single-pass; we re-instantiate via re-iteration.
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            if mvn is not None:
                features = mvn(features, feature_lengths)
            if spec_aug is not None and model.training:
                features = spec_aug(features, feature_lengths)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = raw_model(
                    features, feature_lengths, targets=tokens, target_lengths=target_lengths
                )
                loss = out.loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * grad_accum_steps
            accum_count += 1
            accum_batch_utt += int(features.size(0))

            if accum_count % grad_accum_steps != 0:
                continue

            # ── Optimizer step ──
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), max_grad_norm
            ).item()
            clipped = grad_norm > max_grad_norm
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Health bookkeeping — fold all length stats into a single device→host sync.
            cur_lr = optimizer.param_groups[0]["lr"]
            stats = torch.stack([
                feature_lengths.sum(), feature_lengths.max(),
                target_lengths.sum(), target_lengths.max(),
            ]).cpu().tolist()
            input_lens_sum, input_lens_max, target_lens_sum, target_lens_max = (int(x) for x in stats)
            batch_frames = input_lens_sum
            batch_tokens = target_lens_sum
            total_grid = features.size(0) * features.size(1)
            pad_frames = total_grid - input_lens_sum
            pad_ratio = pad_frames / max(1, total_grid)

            health.step_update(
                loss=accum_loss / max(1, accum_count),
                grad_norm=grad_norm,
                clipped=clipped,
                lr=cur_lr,
                batch_size=int(features.size(0)),
                batch_frames=batch_frames,
                batch_tokens=batch_tokens,
                input_lens_sum=input_lens_sum,
                input_lens_max=input_lens_max,
                target_lens_sum=target_lens_sum,
                target_lens_max=target_lens_max,
                pad_frames_ratio=pad_ratio,
                effective_batch_utterances=accum_batch_utt,
            )

            if tb_writer is not None and is_main_process():
                tb_writer.add_scalar("train/loss", accum_loss / max(1, accum_count), step)
                tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                tb_writer.add_scalar("train/lr", cur_lr, step)

            accum_loss = 0.0
            accum_count = 0
            accum_batch_utt = 0

            # ── Health report ──
            if is_main_process() and (step % hcfg.interval_steps == 0):
                health.emit_report(step=step, max_steps=max_steps)

            # ── Save latest checkpoint ──
            if is_main_process() and (step % save_every_steps == 0):
                t0 = time.time()
                save_kwargs = dict(
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if scaler.is_enabled() else None,
                    step=step,
                    epoch=epoch,
                    best_wer=validator.best_wer,
                    config=cfg,
                )
                save_checkpoint(output_dir / "checkpoints" / "latest.pt", **save_kwargs)
                save_checkpoint(output_dir / "checkpoints" / f"checkpoint-{step:07d}.pt", **save_kwargs)
                # Prune rolling
                rolling_files = sorted((output_dir / "checkpoints").glob("checkpoint-*.pt"))
                for old in rolling_files[:-rolling_keep]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
                health.add_save_time(time.time() - t0)

            if step >= max_steps:
                break

        # ── End of epoch — validation ──
        if (epoch % validate_every_epochs == 0) and is_main_process():
            t0 = time.time()
            snap = validator.run(
                raw_model,
                valid_loader,
                step=step,
                epoch=epoch,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            health.add_validation_time(time.time() - t0)
            health.record_benchmark(snap)

            # Top-K best
            top_k_mgr.consider(
                step=step,
                wer=snap.wer,
                model=raw_model,
                config=cfg,
            )

            # Plots
            t1 = time.time()
            render_validation_plots(
                history=validator.history,
                lr_history=list(health.lr_history),
                output_dir=output_dir,
            )
            health.add_visualization_time(time.time() - t1)

            # TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("valid/wer", snap.wer, step)
                tb_writer.add_scalar("valid/cer", snap.cer, step)
                tb_writer.add_scalar("valid/loss", snap.loss, step)
                last_row = validator.history[-1]
                for r in last_row.get("per_pass") or []:
                    tb_writer.add_scalar(f"valid/wer/m={r['m']}", r["wer"], step)
                    tb_writer.add_scalar(f"valid/cer/m={r['m']}", r["cer"], step)
                    tb_writer.add_scalar(f"valid/loss/m={r['m']}", r["loss"], step)

    if is_main_process():
        logger.info("Training complete.")
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    teardown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
