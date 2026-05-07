from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

PAIR_SCORING_ARTIFACT_MODELS = {
    "TwoTowerRecommender": {
        "retrieval_model_state_dict",
        "ranking_model_state_dict",
        "retrieval_model_kwargs",
        "ranking_model_kwargs",
        "feature_store",
        "config",
    },
    "TwoTowerV2Recommender": {
        "model_state_dict",
        "model_kwargs",
        "features",
        "config",
    },
    "YouTubeTwoTowerRecommender": {
        "model_state_dict",
        "model_kwargs",
        "features",
        "config",
    },
}


def cpu_state_dict(model: nn.Module | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict() if isinstance(model, nn.Module) else model
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def torch_model_artifact(
    model: nn.Module,
    model_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_version": 1,
        "model": model.__class__.__name__,
        "model_state_dict": cpu_state_dict(model),
    }
    if model_kwargs is not None:
        artifact["model_kwargs"] = model_kwargs
    if config is not None:
        artifact["config"] = config
    if metadata is not None:
        artifact["metadata"] = metadata
    return artifact


def save_model_artifact(
    model_dir: str,
    experiment: str,
    artifact: Any,
) -> Path:
    out = Path(model_dir) / f"{experiment}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(artifact, nn.Module):
        artifact = torch_model_artifact(artifact)
    torch.save(artifact, out)
    print(f"Model artifact -> {out}")
    return out


def load_model_artifact(
    model_path: str,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return torch.load(path, map_location=map_location, weights_only=weights_only)


def require_artifact_fields(
    artifact: Any,
    required_fields: set[str],
    context: str = "Model artifact",
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError(f"{context} must be a dict")
    missing = required_fields - set(artifact)
    if missing:
        raise ValueError(f"{context} is missing required fields: {sorted(missing)}")
    return artifact


def load_pair_scoring_artifact(model_path: str, map_location: str = "cpu") -> dict[str, Any]:
    artifact = load_model_artifact(model_path, map_location=map_location, weights_only=False)
    artifact = require_artifact_fields(artifact, {"model"}, "Model artifact")
    model_name = artifact.get("model")
    if model_name in {"RandomRecommender", "PopularityRecommender"}:
        raise ValueError(f"{model_name} artifacts are not supported for pair scoring")
    if model_name not in PAIR_SCORING_ARTIFACT_MODELS:
        raise ValueError(
            f"Unsupported model artifact: {model_name!r}. "
            f"Supported models: {sorted(PAIR_SCORING_ARTIFACT_MODELS)}"
        )

    return require_artifact_fields(
        artifact,
        PAIR_SCORING_ARTIFACT_MODELS[model_name],
        "Pair scoring artifact",
    )
