"""
Dependency Injection

FastAPI dependencies for accessing shared resources.
"""

from fastapi import Request

from agave_vision.config.loader import ConfigLoader
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager


def get_model(request: Request) -> YOLOInference:
    """
    Get YOLO inference model from application state.

    Args:
        request: FastAPI request object

    Returns:
        YOLOInference instance
    """
    return request.app.state.model


def get_roi_manager(request: Request) -> ROIManager:
    """
    Get ROI manager from application state.

    Args:
        request: FastAPI request object

    Returns:
        ROIManager instance
    """
    return request.app.state.roi_manager


def get_config_loader(request: Request) -> ConfigLoader:
    """
    Get config loader from application state.

    Args:
        request: FastAPI request object

    Returns:
        ConfigLoader instance
    """
    return request.app.state.config_loader
