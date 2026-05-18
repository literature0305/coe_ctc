# CLAUDE.md — design notes, SOTA targets, debugging guide

This document is for **future Claude sessions** working on this repository.
It is intentionally terse and assumes Phase 1–5 of the implementation plan at
`/root/.claude/plans/binary-sauteeing-cocke.md` are the source of truth.

---

## 1. Project goal

SOTA CTC ASR baseline for an AI conference paper (Chain-of-Encoders / CoE).
Train and evaluate on LibriSpeech 100h / 960h / libri-light, matching or
beating the published numbers in section 2 below.

The CTC head is the **only** training objective in this repo (no joint
attention loss, no hybrid). That is deliberate — CoE downstream work needs a
clean, reproducible CTC baseline first.

---

## 2. SOTA target table (must be beaten or matched)

LibriSpeech **960h**, no external LM:

| Architecture       | Params | dev-clean | dev-other | test-clean | test-other | Source                               |
| ------------------ | -----: | --------: | --------: | ---------: | ---------: | ------------------------------------ |
| Transformer-CTC S  |   30M  |   3.8 %   |   9.5 %   |   4.0 %    |   9.7 %    | ESPNet recipe                        |
| Conformer-CTC S    |   30M  |   3.4 %   |   8.4 %   |   3.7 %    |   8.5 %    | ESPNet conformer_ctc_small           |
| Conformer-CTC M    |  120M  |   2.5 %   |   6.1 %   |   2.6 %    |   6.0 %    | IceFall conformer-ctc                |
| Conformer-CTC L    |  300M  |   2.0 %   |   5.0 %   |   2.1 %    |   4.9 %    | NeMo Conformer-CTC-Large             |
| Zipformer-CTC L    |  300M  |   1.9 %   |   4.6 %   |   2.0 %    |   4.5 %    | IceFall zipformer-ctc                |
| Zipformer-CTC XL   |  800M  |  ~1.85 %  |  ~4.3 %   |  ~1.9 %    |  ~4.2 %    | Our target (no public reference)     |
| Zipformer-CTC L + 4g LM | 300M | 1.7 % | 4.2 %  | 1.9 %      | 4.1 %      | IceFall + LibriSpeech 4-gram pruned  |

LibriSpeech **100h** (train-clean-100 subset of the 960h prep), no LM:

| Architecture     | Params | test-clean | test-other | Source              |
| ---------------- | -----: | ---------: | ---------: | ------------------- |
| Conformer-CTC S  |   30M  |   6.5 %    |  17.3 %    | ESPNet 100h subset  |
| Conformer-CTC M  |  120M  |   5.4 %    |  14.5 %    | IceFall 100h subset |

Each release of this repo must regenerate the four split numbers (dev/test ×
clean/other) and report **CER, WER, latency** (ms/utterance and RTF). The
evaluation script already does this — see `scripts/evaluation/run_evaluation.sh`.

---

## 3. Implementation phases

Tracked at `/root/.claude/plans/binary-sauteeing-cocke.md`. The user requested
phased implementation (3–5 turns). At any point in a future session, check
which Phase artifacts are present:

| Phase | Key files                                                                                    |
| ----- | -------------------------------------------------------------------------------------------- |
| 1     | `pyproject.toml`, `README.md`, `CLAUDE.md`, `src/coe_ctc/{utils,data}/…`, preprocess script  |
| 2     | `src/coe_ctc/models/{subsampling,attention,transformer,conformer,zipformer,ctc,builder}.py`, dataloader |
| 3     | `src/coe_ctc/training/{train,health,validation,visualize,checkpoint}.py`, train.sh, configs   |
| 4     | `src/coe_ctc/decoding/`, `src/coe_ctc/lm/`, evaluation scripts, ngram training                |
| 5     | All Zipformer/Transformer configs, full README polish, integration dry-run                     |

Each phase concludes by running the verification step described in the plan.

---

## 4. Design invariants (do not break)

1. **CTC-only objective**. No joint CE / attention loss. If experiments call
   for hybrid losses, they belong in a sibling branch, not in `train.py`.
2. **fBank-80 features, 10ms shift, 25ms window, no CMVN file** — global mean
   /var is computed online via Lhotse `GlobalMVN` *only when explicitly enabled
   in config*. Default: per-utterance mean-normalization.
3. **1/4 subsampling** via two 3×3 stride-2 conv layers (`Conv2dSubsampling`),
   identical to icefall.
4. **Flash attention**: prefer `F.scaled_dot_product_attention` with
   `enable_flash=True`; fall back to memory-efficient or math kernel on Turing.
5. **DDP only** (no FSDP) — 800M still fits in A100 80GB with bf16. If model
   crosses 1B, revisit.
6. **bf16 on A100/H100**, fp32 on 2080 Ti (mixed-precision via AMP autocast).
7. **`num_workers` cap**: every multiprocessing call respects the user-supplied
   `--num-workers` (default 8). The user explicitly warned that grabbing all
   CPUs crashes the server.
8. **Resume must be exact**: model weights + optimizer + scheduler + step + RNG
   states + best-WER tracker. Use `torch.save` with a single dict.

---

## 5. Health report parity

The TRAINING HEALTH REPORT printed every `health_report_interval` steps **must
visually match** `tsm-trainer008_aed/errlog019-2_output_example`. Concretely:

- Box width 74 chars, Unicode box-drawing characters `┌─┐│└─┘├─┤`.
- Section order: Loss Analysis → Gradient Health → LR → GPU Memory →
  Throughput → Time Breakdown → Data Pipeline → Dataset Statistics.
- Loss Analysis includes: current, EMA fast (α=0.1), EMA slow (α=0.01), best,
  trend %, variance, convergence rate per 1K steps.
- BENCHMARK EVALUATION block uses outer ╔═╗ borders (not ┌─┐), printed during
  validation only.

The implementation lives in `src/coe_ctc/training/health.py` and mirrors the
class structure of `tsm-trainer008_aed/.../train_chronos2.py:1600-1852`
(`ComprehensiveTrainingCallback._make_health_report`) but emits ASR-relevant
metrics (CTC loss instead of forecast/recon, WER/CER instead of WQL/MASE).

---

## 6. Debugging tips

- **NaN loss in first 500 steps** → almost always SpecAugment masking too
  aggressively, or warmup too short. Drop `time_mask_max_frames` to 30 and
  bump `warmup_steps` to 10000.
- **All-blank predictions** → CTC blank-collapse. Check `vocab_size` matches
  BPE model and `blank_idx == 0`. Also verify subsampling factor: feature
  length must exceed transcript length.
- **`scaled_dot_product_attention` slow on 2080 Ti** → expected; Flash is
  unavailable on sm_75. Set `attn_implementation: "math"` in YAML for clean
  fp32 runs.
- **Hostfile mode hangs** → `train.sh` infers MASTER_ADDR via socket; ensure
  the cluster resolves the first hostfile entry. Set `MASTER_ADDR` env-var
  manually to override.
- **`--mode job` does nothing** → the shim writes to `.ssub_jobs/<jobname>.sh`
  for sites without a real ssub binary. Inspect it with `cat`. The real ssub
  binary at `/group-volume/share/space-cli/space` is required for actual
  submission.
- **Preprocess `[skip-extract]` after only one split was extracted** → was a
  bug where `_extract_tar` checked only `downloads/corpora/LibriSpeech/` (the
  parent dir, which goes non-empty after the first split). Fixed by passing
  a per-split `expected_subdir` (e.g. `LibriSpeech/dev-other`) so each
  tarball decides its own cache hit. If you see this skip on a fresh prep,
  delete only the wrongly-skipped split dir and re-run; downloads stay
  cached.
- **`ImportError: cannot import name 'prepare_libri_light'`** → Lhotse 1.x
  renamed the recipe to `prepare_librilight`. `librispeech.py` imports the
  new spelling first and falls back to the old one. If a future Lhotse drop
  renames it again, update the try/except in `prepare_manifests`.

---

## 7. Useful pointers from this session

- Reference output to mimic: `/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer008_aed/tsm-trainer/errlog019-2_output_example`
- Reference shell script structure: `/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer008_aed/tsm-trainer/scripts/forecasting/training/train.sh`
- Reference health-report class: `tsm-trainer008_aed/.../train_chronos2.py:1600-1852`
- Reference job submission CLI: `tsm-trainer008_aed/.../scripts/forecasting/training/utils/space_job_submission.py`
- IceFall library (model design reference): https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR
- LibriSpeech LM ARPAs / corpus: https://www.openslr.org/11

---

## 8. Out of scope (explicitly)

- Streaming / chunked attention (offline only).
- RNN-T / transducer heads.
- Self-supervised pretraining (wav2vec2, HuBERT).
- WFST decoding via k2 — optional via `[k2]` extra, but the default path uses
  `pyctcdecode` for portability.
- Joint CTC/attention training (paper studies CTC-only baseline).
