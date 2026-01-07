"""
Model Configuration Loader

Centralized model configuration management.
All model paths should be loaded from this module to enable easy updates.
"""

from pathlib import Path
from typing import Optional

import yaml

from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class ModelConfig:
    """
    Model configuration singleton.

    Loads model configuration from configs/model.yaml and provides
    centralized access to model paths across the project.

    Example:
        >>> from agave_vision.config.model_config import get_default_model_path
        >>> model_path = get_default_model_path()
        >>> print(model_path)  # "models/agave-industrial-vision-v1.0.0.pt"
    """

    _instance: Optional["ModelConfig"] = None
    _config: Optional[dict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load model configuration from YAML file."""
        config_path = Path("configs/model.yaml")

        if not config_path.exists():
            logger.warning(
                f"Model config not found at {config_path}. "
                "Using default: models/agave-industrial-vision-v1.0.0.pt"
            )
            self._config = {
                "default_model": "models/agave-industrial-vision-v1.0.0.pt",
                "inference": {
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "image_size": 640,
                    "device": "cpu"
                }
            }
            return

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        logger.debug(f"Loaded model config: {self._config.get('default_model')}")

    @property
    def default_model_path(self) -> str:
        """Get default model path from config."""
        return self._config.get("default_model", "models/agave-industrial-vision-v1.0.0.pt")

    @property
    def model_info(self) -> dict:
        """Get model metadata."""
        return self._config.get("model_info", {})

    @property
    def inference_defaults(self) -> dict:
        """Get default inference parameters."""
        return self._config.get("inference", {})

    def get_version_path(self, version: str) -> Optional[str]:
        """
        Get model path for specific version.

        Args:
            version: Version string (e.g., "v1.0.0")

        Returns:
            Model path or None if version not found
        """
        versions = self._config.get("versions", {})
        version_info = versions.get(version)
        if version_info:
            return version_info.get("path")
        return None


# Singleton instance
_model_config = ModelConfig()


def get_default_model_path() -> str:
    """
    Get the default model path from configuration.

    This is the recommended way to get the model path throughout the project.

    Returns:
        Path to default model (e.g., "models/agave-industrial-vision-v1.0.0.pt")

    Example:
        >>> from agave_vision.config.model_config import get_default_model_path
        >>> model_path = get_default_model_path()
        >>> ml = AgaveVisionML(model_path=model_path)
    """
    return _model_config.default_model_path


def get_model_info() -> dict:
    """
    Get model metadata.

    Returns:
        Dictionary with model information (name, version, classes, etc.)
    """
    return _model_config.model_info


def get_inference_defaults() -> dict:
    """
    Get default inference parameters.

    Returns:
        Dictionary with default confidence, IOU thresholds, etc.
    """
    return _model_config.inference_defaults


def get_model_path_for_version(version: str) -> Optional[str]:
    """
    Get model path for a specific version.

    Args:
        version: Version string (e.g., "v1.0.0")

    Returns:
        Model path or None if version not found
    """
    return _model_config.get_version_path(version)
