"""fBank-80 feature extraction.

We use Lhotse's ``Fbank`` extractor with Kaldi-compatible parameters and write
features as ``LilcomChunkyWriter`` shards for fast random access at train time.
The number of parallel feature-extraction jobs is capped by the caller's
``num_workers`` argument (default 8) so we never overwhelm a shared server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class FbankConfig:
    """Static fBank parameters — matches IceFall LibriSpeech defaults."""

    num_mel_bins: int = 80
    sampling_rate: int = 16000
    frame_length: float = 0.025  # 25 ms
    frame_shift: float = 0.010   # 10 ms
    low_freq: float = 20.0
    high_freq: float = -400.0    # Kaldi convention: high_freq = sample_rate - 400
    snip_edges: bool = False
    use_energy: bool = False
    remove_dc_offset: bool = True
    preemphasis_coefficient: float = 0.97

    # Output sharding — features are stored via ``LilcomChunkyWriter`` when
    # the optional ``lilcom`` package is installed, else ``NumpyFilesWriter``.
    shard_seconds: float = 1800.0        # ~30 min per shard keeps file count manageable

    def to_lhotse_kwargs(self) -> dict:
        # Lhotse's FbankConfig spells the pre-emphasis knob ``preemph_coeff``;
        # older revisions used ``preemphasis_coefficient``. We keep the public
        # field human-readable and rename only when handing off.
        return {
            "num_mel_bins": self.num_mel_bins,
            "sampling_rate": self.sampling_rate,
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "snip_edges": self.snip_edges,
            "remove_dc_offset": self.remove_dc_offset,
            "preemph_coeff": self.preemphasis_coefficient,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fbank(
    manifest_dir: str | Path,
    feats_dir: str | Path,
    *,
    dataset_parts: Iterable[str] | None = None,
    num_workers: int = 8,
    config: FbankConfig | None = None,
    prefix: str = "librispeech",
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Extract fBank-80 features for each part in *dataset_parts*.

    Reads ``{prefix}_recordings_{part}.jsonl.gz`` + ``{prefix}_supervisions_{part}.jsonl.gz``
    from *manifest_dir*, writes feature shards to *feats_dir*, and
    emits a cut set ``cuts_{part}.jsonl.gz`` back into *manifest_dir* with
    feature references already wired up.

    Returns: mapping ``part → cuts_path``.
    """
    log = log or logger
    cfg = config or FbankConfig()
    manifest_dir = Path(manifest_dir)
    feats_dir = Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports: heavy.
    from lhotse import CutSet, Fbank, FbankConfig as LhotseFbankConfig
    from lhotse import RecordingSet, SupervisionSet

    # LilcomChunkyWriter needs the optional ``lilcom`` package; fall back to
    # NumpyFilesWriter (uncompressed, ~4× larger on disk) if it's missing.
    try:
        from lhotse import LilcomChunkyWriter
        import lilcom  # noqa: F401  -- presence check
        storage_writer = LilcomChunkyWriter
    except ImportError:
        from lhotse import NumpyFilesWriter
        log.warning("lilcom not installed — falling back to NumpyFilesWriter (uncompressed).")
        storage_writer = NumpyFilesWriter

    extractor = Fbank(LhotseFbankConfig(**cfg.to_lhotse_kwargs()))

    parts = list(dataset_parts) if dataset_parts is not None else _discover_parts(manifest_dir, prefix)

    out: dict[str, Path] = {}

    for part in parts:
        rec_path = manifest_dir / f"{prefix}_recordings_{part}.jsonl.gz"
        sup_path = manifest_dir / f"{prefix}_supervisions_{part}.jsonl.gz"
        if not rec_path.exists() or not sup_path.exists():
            log.warning(f"  [skip] missing manifests for part {part}: {rec_path} / {sup_path}")
            continue

        cuts_out = manifest_dir / f"cuts_{part}.jsonl.gz"
        if cuts_out.exists():
            log.info(f"  [skip-feats] {cuts_out.name} already exists.")
            out[part] = cuts_out
            continue

        log.info(f"Building cut set for part '{part}'.")
        recs = RecordingSet.from_jsonl_lazy(str(rec_path))
        sups = SupervisionSet.from_jsonl_lazy(str(sup_path))
        cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)

        # Trim each cut to a single supervision (utterance-level CTC).
        # ``compute_and_store_features`` needs ``__len__``, so materialize here.
        cuts = cuts.trim_to_supervisions(keep_overlapping=False).to_eager()

        log.info(f"  Extracting fBank-80 features (num_jobs={num_workers}).")
        storage_path = feats_dir / f"{prefix}_feats_{part}"
        cuts_with_feats = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=str(storage_path),
            num_jobs=max(1, int(num_workers)),
            storage_type=storage_writer,
        )

        cuts_with_feats.to_file(str(cuts_out))
        log.info(f"  → {cuts_out}")
        out[part] = cuts_out

    return out


def _discover_parts(manifest_dir: Path, prefix: str) -> list[str]:
    """Infer dataset parts from recording manifest filenames present on disk."""
    parts: list[str] = []
    for p in manifest_dir.glob(f"{prefix}_recordings_*.jsonl.gz"):
        name = p.name
        head = f"{prefix}_recordings_"
        tail = ".jsonl.gz"
        if name.startswith(head) and name.endswith(tail):
            parts.append(name[len(head) : -len(tail)])
    return sorted(parts)


# ---------------------------------------------------------------------------
# Standalone CLI (rarely used; the run_preprocess.sh wraps everything)
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Extract fBank-80 features for a Lhotse manifest dir.")
    p.add_argument("--manifest-dir", required=True, help="Directory with {prefix}_recordings_*.jsonl.gz.")
    p.add_argument("--feats-dir", required=True, help="Output directory for feature shards.")
    p.add_argument("--prefix", default="librispeech", help="Manifest filename prefix (default: librispeech).")
    p.add_argument("--parts", nargs="*", default=None, help="Parts to process; default = all discovered.")
    p.add_argument("--num-workers", type=int, default=8, help="Parallel feature-extraction jobs.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    compute_fbank(
        manifest_dir=args.manifest_dir,
        feats_dir=args.feats_dir,
        prefix=args.prefix,
        dataset_parts=args.parts,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
