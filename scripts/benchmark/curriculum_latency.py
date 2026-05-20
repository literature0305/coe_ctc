"""Benchmark: latency overhead introduced by curriculum-learning batch construction.

Three measurements:

  A. Per-step  : every micro-step calls `_loader_cap(step)` →
                 `curriculum.max_seconds_for(step)` → linear scan of
                 `cl_schedule_step`. Negligible per call, but it runs every
                 micro-step for the entire training run.

  B. Initial   : one-time loader build at training start.
                 (Cold cache: read JSONL → materialize eager CutSet → filter
                 → eager → sampler → DataLoader.)

  C. Per-phase : loader rebuild at each curriculum boundary.
                 (Warm cache: filter eager CutSet → eager → sampler →
                 DataLoader. No JSONL re-parse, thanks to `_eager_cuts_cache`.)

The script uses a synthetic CutSet so we can run anywhere (no manifests, no
GPU, no audio data needed). Cuts have realistic LibriSpeech-100h-like
duration distribution: log-normal in [1, 20]s, mean ≈ 12s.

Usage:
    python scripts/benchmark/curriculum_latency.py
    python scripts/benchmark/curriculum_latency.py --num-cuts 280000   # libri-960h scale
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import numpy as np
from lhotse import CutSet, MonoCut
from lhotse.audio import Recording, AudioSource
from lhotse.supervision import SupervisionSegment

from coe_ctc.training.curriculum import CurriculumConfig, parse_curriculum_config


def _make_synthetic_cuts(num_cuts: int, seed: int = 0) -> CutSet:
    """Build a CutSet of `num_cuts` MonoCuts whose durations mimic LibriSpeech."""
    rng = np.random.default_rng(seed)
    # log-normal centered around 12s, clipped to [1, 20]
    raw = rng.lognormal(mean=np.log(8.0), sigma=0.5, size=num_cuts)
    durations = np.clip(raw, 1.0, 20.0)

    cuts = []
    for i, dur in enumerate(durations):
        rec = Recording(
            id=f"rec-{i:07d}",
            sources=[AudioSource(type="file", channels=[0], source="/dev/null")],
            sampling_rate=16000,
            num_samples=int(float(dur) * 16000),
            duration=float(dur),
        )
        sup = SupervisionSegment(
            id=f"sup-{i:07d}",
            recording_id=rec.id,
            start=0.0,
            duration=float(dur),
            channel=0,
            text="a b c",
        )
        cuts.append(MonoCut(
            id=f"cut-{i:07d}",
            start=0.0,
            duration=float(dur),
            channel=0,
            recording=rec,
            supervisions=[sup],
        ))
    return CutSet.from_cuts(cuts)


def _bench(fn: Callable[[], object], *, repeats: int = 5, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_ms": statistics.median(samples) * 1000,
        "min_ms": min(samples) * 1000,
        "max_ms": max(samples) * 1000,
        "samples": [s * 1000 for s in samples],
    }


def benchmark_per_step(curriculum: CurriculumConfig, max_seconds: float, n_calls: int = 5_000_000) -> dict:
    """Measure `max_seconds_for(step)` overhead per micro-step."""
    t0 = time.perf_counter()
    for s in range(n_calls):
        curriculum.max_seconds_for(s % 6000, fallback=max_seconds)
    elapsed = time.perf_counter() - t0
    return {
        "calls": n_calls,
        "total_sec": elapsed,
        "ns_per_call": elapsed / n_calls * 1e9,
    }


def benchmark_loader_build(cuts: CutSet, max_seconds: float | None) -> dict:
    """Time the filter → to_eager → sampler-construct path used on rebuild."""
    from lhotse.dataset.sampling import DynamicBucketingSampler

    def _build():
        cap = max_seconds if max_seconds is not None else 20.0
        filt = cuts.filter(lambda c: 1.0 <= c.duration <= cap).to_eager()
        sampler = DynamicBucketingSampler(
            filt, shuffle=True, drop_last=True, max_duration=100.0, num_buckets=30,
        )
        # Force sampler bucket prep so the timing is comparable to real init.
        _ = len(sampler) if hasattr(sampler, "__len__") else None
        return filt, sampler

    return _bench(_build, repeats=5, warmup=1)


def benchmark_initial_build(cuts: CutSet, max_seconds: float) -> dict:
    """Cold-cache path: JSONL → eager → filter → eager → sampler.

    Writes the cut set out to a tempfile and re-reads it via
    `CutSet.from_jsonl_lazy`, mimicking what `_eager_cuts` does on its
    very first call (or what the OLD code did on every phase transition
    before the `_eager_cuts_cache` was added).
    """
    import tempfile, os
    from lhotse.dataset.sampling import DynamicBucketingSampler

    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False)
    tmp.close()
    cuts.to_jsonl(tmp.name)

    def _build():
        eager = CutSet.from_jsonl_lazy(tmp.name).to_eager()
        filt = eager.filter(lambda c: 1.0 <= c.duration <= max_seconds).to_eager()
        sampler = DynamicBucketingSampler(
            filt, shuffle=True, drop_last=True, max_duration=100.0, num_buckets=30,
        )
        _ = len(sampler) if hasattr(sampler, "__len__") else None
        return filt, sampler

    try:
        return _bench(_build, repeats=3, warmup=0)
    finally:
        os.unlink(tmp.name)


def _fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms/1000:.3f} s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-cuts", type=int, default=28000,
                   help="Cuts in the synthetic set. libri100 ≈ 28k, libri960 ≈ 280k.")
    p.add_argument("--max-steps", type=int, default=600_000,
                   help="Total training steps, used to amortize per-step overhead.")
    p.add_argument("--per-step-calls", type=int, default=5_000_000,
                   help="Number of `_loader_cap` calls to time.")
    p.add_argument("--schedule-len", type=float, nargs="+", default=[2, 4, 8])
    p.add_argument("--schedule-step", type=int, nargs="+", default=[1000, 2000, 4000])
    args = p.parse_args()

    print("=" * 78)
    print("Curriculum batch-construction latency benchmark")
    print("=" * 78)
    print(f"Cuts:                {args.num_cuts:,}  (libri100≈28k, libri960≈280k)")
    print(f"Schedule:            cl_schedule_len={args.schedule_len}  "
          f"cl_schedule_step={args.schedule_step}")
    print(f"Training horizon:    {args.max_steps:,} steps")
    print()

    curriculum = parse_curriculum_config({
        "cl_schedule_len": args.schedule_len,
        "cl_schedule_step": args.schedule_step,
    })
    fallback_max_sec = 17.0
    curriculum.validate(min_seconds=1.0, max_seconds=fallback_max_sec)

    print("Building synthetic cuts...", flush=True)
    t0 = time.perf_counter()
    cuts = _make_synthetic_cuts(args.num_cuts)
    print(f"  built {len(cuts):,} cuts in {time.perf_counter()-t0:.2f}s\n")

    # ─── A. Per-step overhead ────────────────────────────────────────────
    print("─" * 78)
    print("A. Per-step  `max_seconds_for(step)` overhead")
    print("─" * 78)
    a = benchmark_per_step(curriculum, fallback_max_sec, n_calls=args.per_step_calls)
    total_amortized_ms = a["ns_per_call"] * args.max_steps / 1e6
    print(f"  calls timed       : {a['calls']:,}")
    print(f"  wall time         : {a['total_sec']:.3f} s")
    print(f"  per call          : {a['ns_per_call']:.1f} ns")
    print(f"  over {args.max_steps:,} steps : ~{total_amortized_ms:.1f} ms total")
    print()

    # ─── B. Initial loader build (cold cache) ────────────────────────────
    print("─" * 78)
    print("B. Initial loader build  (cold cache: JSONL→eager + filter→eager + sampler)")
    print("─" * 78)
    # Phase-0 cap = schedule_len[0]
    b = benchmark_initial_build(cuts, max_seconds=args.schedule_len[0])
    print(f"  median            : {_fmt_ms(b['median_ms'])}")
    print(f"  min / max         : {_fmt_ms(b['min_ms'])} / {_fmt_ms(b['max_ms'])}")
    print(f"  samples (ms)      : {', '.join(f'{s:.1f}' for s in b['samples'])}")
    print()
    print("  NOTE: this cost is paid once at training start, with or without curriculum.")
    print("  Without curriculum it uses cap = data.max_seconds (slightly larger filter).")
    print()

    # ─── C. Per-phase rebuild (warm cache) ───────────────────────────────
    print("─" * 78)
    print("C. Per-phase loader rebuild  (warm eager cache: filter→eager + sampler)")
    print("─" * 78)
    # The eager cache is populated; we time only the filter+eager+sampler step.
    eager_cache = cuts.to_eager()  # mimic populated `_eager_cuts_cache`
    per_phase_results = []
    for phase_idx, cap in enumerate(args.schedule_len):
        c = benchmark_loader_build(eager_cache, max_seconds=cap)
        per_phase_results.append((cap, c))
        print(f"  phase {phase_idx} cap=≤{cap:g}s : median {_fmt_ms(c['median_ms'])}  "
              f"(min {_fmt_ms(c['min_ms'])}, max {_fmt_ms(c['max_ms'])})")
    # Final transition: cap = data.max_seconds
    c_final = benchmark_loader_build(eager_cache, max_seconds=fallback_max_sec)
    per_phase_results.append((fallback_max_sec, c_final))
    print(f"  phase {len(args.schedule_len)} cap=≤{fallback_max_sec:g}s "
          f": median {_fmt_ms(c_final['median_ms'])}  "
          f"(min {_fmt_ms(c_final['min_ms'])}, max {_fmt_ms(c_final['max_ms'])})")
    print()

    # ─── Total amortized cost ────────────────────────────────────────────
    print("─" * 78)
    print("Total amortized overhead attributable to curriculum")
    print("─" * 78)
    # Curriculum-induced rebuilds: len(schedule) rebuilds (initial + N-1 transitions
    # + one final). Without curriculum, we'd still pay one initial build → so the
    # curriculum-attributable rebuild cost is N transitions (the initial build is
    # counted under any training run).
    transition_rebuilds_ms = sum(r[1]["median_ms"] for r in per_phase_results) - per_phase_results[0][1]["median_ms"]
    per_step_total_ms = a["ns_per_call"] * args.max_steps / 1e6
    total_ms = transition_rebuilds_ms + per_step_total_ms
    print(f"  per-step `max_seconds_for` (×{args.max_steps:,} steps): "
          f"~{per_step_total_ms:.1f} ms")
    print(f"  phase-transition rebuilds (×{len(args.schedule_len)}): "
          f"~{_fmt_ms(transition_rebuilds_ms)}")
    print(f"  -------------------------------------------------------")
    print(f"  total curriculum overhead             : ~{_fmt_ms(total_ms)}")
    print()
    # Compare against typical training wall time.
    # CTC training: ~1 step/s on a 300M Conformer A100x8 → 600k steps ≈ 7 days.
    typical_training_sec = args.max_steps * 1.0
    pct = total_ms / 1000 / typical_training_sec * 100
    print(f"  for context: at ~1 step/s ({args.max_steps:,} steps ≈ "
          f"{typical_training_sec/3600:.1f} h), curriculum is "
          f"{pct:.4f}% of total wall time.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
