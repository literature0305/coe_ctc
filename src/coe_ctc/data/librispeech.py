"""LibriSpeech / libri-light download + Lhotse manifest preparation.

The 100h prep is a strict subset of the 960h prep — calling
``prepare_manifests(name="libri960")`` also writes a ``libri100/`` manifest
directory derived from the ``train-clean-100`` split.

Heavy lifting is delegated to Lhotse's ``prepare_librispeech`` /
``prepare_librilight`` recipes; we only orchestrate. (Older Lhotse versions
spelled the libri-light recipe ``prepare_libri_light``; both are supported.)
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

LIBRISPEECH_SPLITS = {
    "train-clean-100": "train-clean-100.tar.gz",
    "train-clean-360": "train-clean-360.tar.gz",
    "train-other-500": "train-other-500.tar.gz",
    "dev-clean": "dev-clean.tar.gz",
    "dev-other": "dev-other.tar.gz",
    "test-clean": "test-clean.tar.gz",
    "test-other": "test-other.tar.gz",
}

LIBRI_LIGHT_SPLITS = {
    # libri-light is enormous; we list the small/medium subsets by default.
    "small": "https://dl.fbaipublicfiles.com/librilight/data/small.tar",
    "medium": "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
    "large": "https://dl.fbaipublicfiles.com/librilight/data/large.tar",
}

LIBRISPEECH_BASE_URL = "https://www.openslr.org/resources/12"
LIBRISPEECH_LM_BASE_URL = "https://www.openslr.org/resources/11"

# Files served from openslr-11 (the LibriSpeech LM resource page). Keys are
# the logical names users pass via --with-lm / --lm-resources.
LM_RESOURCES: dict[str, dict[str, str]] = {
    "vocab":        {"file": "librispeech-vocab.txt",          "subdir": "lms"},
    "lexicon":      {"file": "librispeech-lexicon.txt",        "subdir": "lms"},
    "corpus":       {"file": "librispeech-lm-norm.txt.gz",     "subdir": "corpus"},
    "3gram":        {"file": "3-gram.arpa.gz",                 "subdir": "lms"},
    "3gram-pruned": {"file": "3-gram.pruned.1e-7.arpa.gz",     "subdir": "lms"},
    "4gram":        {"file": "4-gram.arpa.gz",                 "subdir": "lms"},
}


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a supported dataset preset."""

    name: str
    train_splits: tuple[str, ...]
    dev_splits: tuple[str, ...]
    test_splits: tuple[str, ...]
    source: str  # "librispeech" | "libri_light"
    extra_splits: tuple[str, ...] = field(default_factory=tuple)


def manifest_prefix_for(data: str) -> str:
    """Lhotse filename prefix used by ``cuts_<part>.jsonl.gz`` and friends.

    LibriSpeech corpora (libri100 / libri960) write ``librispeech_*`` files;
    libri-light writes ``libri-light_*``. Centralised so train/eval scripts
    don't each repeat the same string-matching logic.
    """
    spec = DATASETS.get(data)
    if spec is None:
        # Conservative default — bail to LibriSpeech prefix when the alias is
        # unknown so callers see a clean "missing manifest" error downstream
        # rather than a wrong-prefix mismatch.
        return "librispeech"
    return "librispeech" if spec.source == "librispeech" else "libri-light"


DATASETS: dict[str, DatasetSpec] = {
    "libri100": DatasetSpec(
        name="libri100",
        train_splits=("train-clean-100",),
        dev_splits=("dev-clean", "dev-other"),
        test_splits=("test-clean", "test-other"),
        source="librispeech",
    ),
    "libri960": DatasetSpec(
        name="libri960",
        train_splits=("train-clean-100", "train-clean-360", "train-other-500"),
        dev_splits=("dev-clean", "dev-other"),
        test_splits=("test-clean", "test-other"),
        source="librispeech",
        # The 960h prep ALSO produces a 100h-only manifest dir.
        extra_splits=("derived-libri100",),
    ),
    "libri_light": DatasetSpec(
        name="libri_light",
        train_splits=("small",),  # small is the default; configurable via --libri-light-splits
        dev_splits=(),
        test_splits=(),
        source="libri_light",
    ),
}


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------


def _download_one(url: str, dest: Path, *, log: logging.Logger | None = None) -> Path:
    log = log or logger
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"  [skip-download] {dest.name} already present.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info(f"  Downloading {url} → {dest.name}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)  # 1 MiB chunks
    tmp.rename(dest)
    return dest


def _extract_tar(
    archive: Path,
    target_dir: Path,
    *,
    expected_subdir: str | None = None,
    log: logging.Logger | None = None,
) -> Path:
    """Extract *archive* into *target_dir* unless ``expected_subdir`` already exists.

    The old version checked only ``target_dir/LibriSpeech``, which goes
    non-empty as soon as the first split (e.g. dev-clean) is extracted —
    causing every subsequent extract call to be wrongly skipped. Now the
    caller passes a SPLIT-specific subdir so we only skip the work that's
    actually done.
    """
    log = log or logger
    target_dir.mkdir(parents=True, exist_ok=True)
    if expected_subdir is not None:
        split_marker = target_dir / expected_subdir
        if split_marker.exists() and any(split_marker.iterdir()):
            log.info(f"  [skip-extract] {split_marker} already populated.")
            return target_dir
    log.info(f"  Extracting {archive.name} → {target_dir}")
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(target_dir)
    return target_dir


def download_dataset(
    name: str,
    download_dir: str | Path,
    *,
    libri_light_splits: Iterable[str] | None = None,
    log: logging.Logger | None = None,
) -> Path:
    """Download (and extract) the raw corpus for *name* into ``<download_dir>/corpora/``.

    For LibriSpeech this fetches the tarballs listed in
    ``DATASETS[name].{train,dev,test}_splits`` from openslr-12.

    For libri-light we honor the provided ``libri_light_splits`` (else the
    spec's defaults).

    Returns the path to the extracted corpus directory.
    """
    log = log or logger
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    spec = DATASETS[name]

    corpora_root = Path(download_dir) / "corpora"
    corpora_root.mkdir(parents=True, exist_ok=True)

    if spec.source == "librispeech":
        extract_dir = corpora_root  # LibriSpeech tars unpack to ./LibriSpeech/<split>/
        wanted = set(spec.train_splits) | set(spec.dev_splits) | set(spec.test_splits)
        for split in sorted(wanted):
            url = f"{LIBRISPEECH_BASE_URL}/{split}.tar.gz"
            tarball = corpora_root / "tars" / f"{split}.tar.gz"
            _download_one(url, tarball, log=log)
            _extract_tar(tarball, extract_dir, expected_subdir=f"LibriSpeech/{split}", log=log)
        return extract_dir / "LibriSpeech"

    if spec.source == "libri_light":
        ll_root = corpora_root / "libri_light"
        ll_root.mkdir(parents=True, exist_ok=True)
        splits = list(libri_light_splits) if libri_light_splits else list(spec.train_splits)
        for split in splits:
            if split not in LIBRI_LIGHT_SPLITS:
                raise ValueError(
                    f"Unknown libri-light split '{split}'. Available: {list(LIBRI_LIGHT_SPLITS)}"
                )
            url = LIBRI_LIGHT_SPLITS[split]
            tarball = corpora_root / "tars" / f"libri_light-{split}.tar"
            _download_one(url, tarball, log=log)
            # libri-light tars unpack to ``libri_light/<split>/`` directly.
            _extract_tar(tarball, ll_root, expected_subdir=split, log=log)
        return ll_root

    raise RuntimeError(f"Unhandled dataset source '{spec.source}'.")


# ---------------------------------------------------------------------------
# Manifest preparation (delegates to Lhotse)
# ---------------------------------------------------------------------------


def prepare_manifests(
    name: str,
    corpus_dir: str | Path,
    manifest_dir: str | Path,
    *,
    num_workers: int = 8,
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Build Lhotse RecordingSet/SupervisionSet manifests for *name*.

    For the LibriSpeech path we use ``lhotse.recipes.prepare_librispeech``.

    For the 960h prep we additionally derive a ``libri100/`` manifest dir
    containing only the ``train-clean-100`` cut, so downstream training can
    point at either alias.

    Returns a mapping of split-name → manifest jsonl.gz path.
    """
    log = log or logger
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'.")
    spec = DATASETS[name]
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(corpus_dir)

    # Lazy import — heavy dependency, only needed at preprocess time.
    # Lhotse renamed ``prepare_libri_light`` → ``prepare_librilight`` somewhere
    # in the 1.x line; accept either spelling.
    from lhotse.recipes import prepare_librispeech
    try:
        from lhotse.recipes import prepare_librilight
    except ImportError:
        from lhotse.recipes import prepare_libri_light as prepare_librilight

    out: dict[str, Path] = {}

    if spec.source == "librispeech":
        dataset_parts = sorted(set(spec.train_splits) | set(spec.dev_splits) | set(spec.test_splits))

        # Fast path: if every part's recordings+supervisions are already on
        # disk, skip the (still relatively slow) Lhotse call entirely.
        all_present = all(
            (manifest_dir / f"librispeech_recordings_{part}.jsonl.gz").exists()
            and (manifest_dir / f"librispeech_supervisions_{part}.jsonl.gz").exists()
            for part in dataset_parts
        )
        if all_present:
            log.info(f"  [skip-manifest] all parts present in {manifest_dir}; using cached manifests.")
        else:
            log.info(f"Calling lhotse.prepare_librispeech (parts={dataset_parts}, num_jobs={num_workers}).")
            prepare_librispeech(
                corpus_dir=str(corpus_dir),
                dataset_parts=dataset_parts,
                output_dir=str(manifest_dir),
                num_jobs=num_workers,
            )
        for part in dataset_parts:
            recs = manifest_dir / f"librispeech_recordings_{part}.jsonl.gz"
            sups = manifest_dir / f"librispeech_supervisions_{part}.jsonl.gz"
            if recs.exists():
                out[f"recordings_{part}"] = recs
            if sups.exists():
                out[f"supervisions_{part}"] = sups

        # Derived 100h-only manifest dir for libri960 → libri100 alias.
        if name == "libri960":
            child = manifest_dir.parent / "libri100"
            child.mkdir(parents=True, exist_ok=True)
            for src in manifest_dir.iterdir():
                if "train-clean-100" in src.name or any(
                    f"_{s}" in src.name for s in spec.dev_splits + spec.test_splits
                ):
                    dst = child / src.name
                    if not dst.exists():
                        _copy_or_symlink(src, dst, log)
            log.info(f"Derived libri100 manifest mirrored at {child}.")

        return out

    if spec.source == "libri_light":
        # libri-light has no canonical part list, so we just check whether
        # ANY libri-light manifest is on disk and skip the (very slow) call.
        existing = list(manifest_dir.glob("libri-light_recordings_*.jsonl.gz"))
        if existing:
            log.info(f"  [skip-manifest] found {len(existing)} libri-light manifests; reusing.")
        else:
            log.info("Calling lhotse.prepare_librilight. This can take a long time on large/medium.")
            prepare_librilight(
                corpus_dir=str(corpus_dir),
                output_dir=str(manifest_dir),
                num_jobs=num_workers,
            )
        for recs in manifest_dir.glob("libri-light_recordings_*.jsonl.gz"):
            part = recs.name[len("libri-light_recordings_"):-len(".jsonl.gz")]
            sups = manifest_dir / f"libri-light_supervisions_{part}.jsonl.gz"
            out[f"recordings_{part}"] = recs
            if sups.exists():
                out[f"supervisions_{part}"] = sups
        return out

    raise RuntimeError(f"Unhandled dataset source '{spec.source}'.")


def _copy_or_symlink(src: Path, dst: Path, log: logging.Logger) -> None:
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        log.info(f"  symlink failed for {dst.name}; falling back to copy.")
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Plain-text transcript dump (used by BPE training)
# ---------------------------------------------------------------------------


_DEFAULT_LM_ITEMS = ("vocab", "3gram-pruned", "4gram")


def download_lm_resources(
    download_dir: str | Path,
    *,
    items: Iterable[str] | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Download LibriSpeech LM resources from openslr-11.

    Args:
        download_dir: Root for ``downloads/`` (gets ``lms/`` and ``corpus/`` children).
        items: Which logical resources to fetch. Defaults to a small set good
            for pyctcdecode N-gram fusion: ``("vocab", "3gram-pruned", "4gram")``.
            Pass ``"all"`` (or list every key) to also pull the lexicon and the
            full 4G corpus.

    Returns: ``{logical_name: Path}``.
    """
    log = log or logger
    items = _normalize_lm_items(items)
    download_dir = Path(download_dir)
    out: dict[str, Path] = {}
    for key in items:
        if key not in LM_RESOURCES:
            raise ValueError(f"Unknown LM resource '{key}'. Available: {list(LM_RESOURCES)}.")
        spec = LM_RESOURCES[key]
        url = f"{LIBRISPEECH_LM_BASE_URL}/{spec['file']}"
        dest = download_dir / spec["subdir"] / spec["file"]
        _download_one(url, dest, log=log)
        out[key] = dest
    return out


def _normalize_lm_items(items: Iterable[str] | str | None) -> tuple[str, ...]:
    """Resolve CLI shorthand (None / "all" / "vocab") into a concrete tuple of keys."""
    if items is None:
        return _DEFAULT_LM_ITEMS
    if isinstance(items, str):
        return tuple(LM_RESOURCES) if items == "all" else (items,)
    return tuple(items)


def load_unigrams(path: str | Path | None) -> list[str] | None:
    """Read a word list from an openslr-11-style file.

    Accepts both ``librispeech-vocab.txt`` (one ``WORD COUNT`` per line) and
    ``librispeech-lexicon.txt`` (one ``WORD PRON1 PRON2 …`` per line) — both
    yield the same set of words (first whitespace-separated token, lowercased).
    Comment lines (``#…``) and blank lines are ignored.

    Returns ``None`` if *path* is falsy so callers can write
    ``unigrams = load_unigrams(args.lexicon)`` without an extra guard.
    """
    if not path:
        return None
    words: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            word = line.split()[0].lower()
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    logger.info(f"Loaded {len(words)} unigrams from {Path(path).name}")
    return words


def dump_transcripts(
    supervision_jsonl_paths: Iterable[str | Path],
    out_txt: str | Path,
    *,
    lowercase: bool = False,
    log: logging.Logger | None = None,
) -> Path:
    """Concatenate transcripts from one or more supervisions manifests into a flat text file.

    The output file is one normalized utterance per line — feed it directly
    to SentencePiece's ``--input`` argument.
    """
    log = log or logger
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import: Lhotse's SupervisionSet handles jsonl.gz transparently.
    from lhotse import SupervisionSet

    written = 0
    with out_txt.open("w", encoding="utf-8") as fh:
        for path in supervision_jsonl_paths:
            log.info(f"  Dumping transcripts from {path}")
            sset = SupervisionSet.from_jsonl_lazy(str(path))
            for seg in sset:
                text = (seg.text or "").strip()
                if not text:
                    continue
                if lowercase:
                    text = text.lower()
                fh.write(text + "\n")
                written += 1
    log.info(f"Wrote {written} transcript lines to {out_txt}.")
    return out_txt
