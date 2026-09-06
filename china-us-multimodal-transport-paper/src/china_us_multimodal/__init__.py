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
from .validation import DataValidationError, validate_model_data

__all__ = [
    "Arc",
    "DataValidationError",
    "EvaluationResult",
    "ModelConfig",
    "ModelData",
    "Mode",
    "Node",
    "NodeKind",
    "RouteAllocation",
    "Service",
    "Shipment",
    "TransferRule",
    "evaluate_solution",
    "load_config",
    "validate_model_data",
]
