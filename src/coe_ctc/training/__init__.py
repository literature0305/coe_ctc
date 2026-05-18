"""Training entry + health report + validation."""

from coe_ctc.training.checkpoint import (
    BestCheckpoint,
    TopKCheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from coe_ctc.utils.distributed import unwrap_model
from coe_ctc.training.health import (
    BenchmarkSnapshot,
    HealthReporterConfig,
    TrainingHealthReporter,
)
from coe_ctc.training.optimizer import OptimConfig, build_optimizer, build_scheduler
from coe_ctc.training.validation import Validator, compute_wer_cer, ctc_greedy_decode
from coe_ctc.training.visualize import render_validation_plots

__all__ = [
    "BenchmarkSnapshot",
    "BestCheckpoint",
    "HealthReporterConfig",
    "OptimConfig",
    "TopKCheckpointManager",
    "TrainingHealthReporter",
    "Validator",
    "build_optimizer",
    "build_scheduler",
    "compute_wer_cer",
    "ctc_greedy_decode",
    "load_checkpoint",
    "render_validation_plots",
    "save_checkpoint",
    "unwrap_model",
]
