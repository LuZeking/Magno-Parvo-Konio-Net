from . import cli
from . import datasets
from . import evaluation
from . import models
from . import plotting
from .model_evaluator import ModelEvaluator
from .plotting.plot import plot
from .version import __version__, VERSION

# Simple evaluator: one-call interface for training-loop / quick evaluation.
from .simple_eval import (
    evaluate,
    visualize,
    compute_shape_bias,
    compute_accuracy,
    register_pytorch_model,
)
from .repr_shape_bias import compute_repr_shape_bias

Evaluate = ModelEvaluator
Plot = plot
