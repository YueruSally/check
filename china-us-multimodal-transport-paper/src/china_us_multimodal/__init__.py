"""Core model for China-US scheduled multimodal freight routing."""

from .config import ModelConfig, load_config
from .domain import (
    Arc,
    ModelData,
    Mode,
    Node,
    NodeKind,
    RouteAllocation,
    Service,
    Shipment,
    TransferRule,
)
from .evaluator import EvaluationResult, evaluate_solution
from .nsga2 import NSGA2Config, NSGA2Result, run_nsga2
from .validation import DataValidationError, validate_model_data

__all__ = [
    "Arc",
    "DataValidationError",
    "EvaluationResult",
    "ModelConfig",
    "ModelData",
    "Mode",
    "NSGA2Config",
    "NSGA2Result",
    "Node",
    "NodeKind",
    "RouteAllocation",
    "Service",
    "Shipment",
    "TransferRule",
    "evaluate_solution",
    "load_config",
    "run_nsga2",
    "validate_model_data",
]
