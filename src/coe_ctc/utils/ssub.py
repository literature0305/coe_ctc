"""ssub job submission shim.

Modeled after ``tsm-trainer008_aed/.../scripts/forecasting/training/utils/space_job_submission.py``.

This is intentionally a SHIM: it constructs the command line that a real
Space/ssub binary would consume, and either:

  * invokes the real binary if found at ``$SSUB_BIN`` or
    ``/group-volume/share/space-cli/space``;
  * writes a launch script under ``.ssub_jobs/<job_name>.sh`` otherwise, so
    sites without ssub can still see exactly what would be submitted.

We do NOT try to authenticate or reach out to any service from this shim;
that's the user's job.

CLI usage:
    python -m coe_ctc.utils.ssub \
        --config scripts/training/configs/conformer_300m_a100x8.yaml \
        --data libri960 --gpu-type A100 --ngpu 8 \
        --train-module coe_ctc.training.train
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="coe_ctc.utils.ssub", description="ssub job submission shim.")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--gpu-type", default="A100", choices=["A100", "H100", "2080ti", "V100"])
    p.add_argument("--ngpu", default="8")
    p.add_argument("--priority", default="3")
    p.add_argument("--exp-id", default="370")
    p.add_argument("--job-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--train-module", default="coe_ctc.training.train",
                   help="The python -m target to invoke on the remote node.")
    p.add_argument("--image", default="sr-asr/coe-ctc-torch2.6.0-cu121:latest")
    p.add_argument("--bucket", default="sdp-asr")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-args", nargs="*", default=[])
    return p.parse_args(argv)


def build_remote_command(args: argparse.Namespace) -> str:
    """Assemble the bash command that the job will execute on the remote node."""
    config_path = Path(args.config).resolve()
    parts = [
        f"cd {shlex.quote(str(_REPO_ROOT))}",
        f"export PYTHONPATH={shlex.quote(str(_REPO_ROOT / 'src'))}:${{PYTHONPATH:-}}",
        "export NCCL_DEBUG=WARN",
        "export OMP_NUM_THREADS=8",
        "export MKL_NUM_THREADS=8",
        "export OPENBLAS_NUM_THREADS=8",
        "export NUMEXPR_MAX_THREADS=8",
        "export TOKENIZERS_PARALLELISM=false",
    ]
    train_cmd = [
        "bash", "scripts/training/train.sh",
        "--config", str(config_path),
    ]
    if args.data:
        train_cmd += ["--data", args.data]
    if args.output_dir:
        train_cmd += ["--output-dir", args.output_dir]
    if args.resume_from_checkpoint:
        train_cmd += ["--resume-from-checkpoint", args.resume_from_checkpoint]
    train_cmd += list(args.extra_args)
    parts.append(" ".join(shlex.quote(a) for a in train_cmd))
    return " && ".join(parts)


def find_ssub_binary() -> str | None:
    """Look up the real ssub binary; return ``None`` if not found."""
    explicit = os.environ.get("SSUB_BIN")
    if explicit and shutil.which(explicit):
        return explicit
    candidates = [
        "/group-volume/share/space-cli/space",
        "ssub",
    ]
    for c in candidates:
        if shutil.which(c) or Path(c).is_file():
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    job_name = args.job_name or f"coe-ctc-{Path(args.config).stem}"

    remote_cmd = build_remote_command(args)
    ssub_cmd = [
        find_ssub_binary() or "ssub",
        f"job_name={job_name}",
        f"cmd={remote_cmd}",
        f"priority={args.priority}",
        f"exp_id={args.exp_id}",
        f"image={args.image}",
        f"ngpu={args.ngpu}",
        f"gpu_type={args.gpu_type}",
        f"bucket={args.bucket}",
    ]

    print("=" * 72)
    print(f"  ssub shim — job_name={job_name}")
    print(f"  gpu_type={args.gpu_type}  ngpu={args.ngpu}  priority={args.priority}")
    print(f"  config={args.config}")
    print(f"  remote_cmd={remote_cmd}")
    print("=" * 72)

    bin_ = find_ssub_binary()
    if bin_ is None or args.dry_run:
        # Write a launch script so the user can inspect exactly what would run.
        out_dir = _REPO_ROOT / ".ssub_jobs"
        out_dir.mkdir(parents=True, exist_ok=True)
        script = out_dir / f"{job_name}.sh"
        with script.open("w", encoding="utf-8") as fh:
            fh.write("#!/usr/bin/env bash\n")
            fh.write("# coe-ctc ssub shim — generated script\n")
            fh.write("# Equivalent ssub command:\n")
            fh.write("#   " + " ".join(shlex.quote(x) for x in ssub_cmd) + "\n\n")
            fh.write(remote_cmd + "\n")
        script.chmod(0o755)
        action = "dry-run" if args.dry_run else "no-ssub"
        print(f"[{action}] wrote {script} (no submission attempted).")
        return 0

    import subprocess
    print(f"Submitting via {bin_} …")
    return subprocess.run(ssub_cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
