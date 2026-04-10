"""Backtest engine: load ``configs/system.yaml`` and expose typed system settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SYSTEM_PATH = _REPO_ROOT / "configs" / "system.yaml"


@dataclass(frozen=True, slots=True)
class SystemConfig:
    latency_seconds: int = 120
    threshold: float = 0.65
    holding_period_hours: float = 6.0
    regime_enabled: bool = True
    slippage_model: str = "gaussian"
    slippage_sigma: float = 1e-4


def _merge_defaults(raw: dict[str, Any]) -> SystemConfig:
    d = {
        "latency_seconds": 120,
        "threshold": 0.65,
        "holding_period_hours": 6.0,
        "regime_enabled": True,
        "slippage_model": "gaussian",
        "slippage_sigma": 1e-4,
    }
    d.update(raw)
    return SystemConfig(
        latency_seconds=int(d["latency_seconds"]),
        threshold=float(d["threshold"]),
        holding_period_hours=float(d["holding_period_hours"]),
        regime_enabled=bool(d["regime_enabled"]),
        slippage_model=str(d["slippage_model"]),
        slippage_sigma=float(d["slippage_sigma"]),
    )


def load_system_config(path: Path | None = None) -> SystemConfig:
    """
    Load YAML from ``configs/system.yaml`` (repo root) unless ``path`` is given.
    Missing file falls back to :class:`SystemConfig` defaults.
    """
    cfg_path = path or _DEFAULT_SYSTEM_PATH
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyYAML is required to load system config (pip install pyyaml)") from e
        with cfg_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"system config must be a mapping, got {type(loaded)}")
            raw = {str(k): v for k, v in loaded.items()}
    return _merge_defaults(raw)


def default_system_config_path() -> Path:
    return _DEFAULT_SYSTEM_PATH
