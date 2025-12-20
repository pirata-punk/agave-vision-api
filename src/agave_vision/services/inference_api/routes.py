"""
API Routes

FastAPI endpoint definitions for the inference API.
"""

import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from agave_vision.config.loader import ConfigLoader
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager

from .dependencies import get_config_loader, get_model, get_roi_manager
from .schemas import CameraROISchema, CameraSchema, HealthResponse, InferResponse, DetectionSchema

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model: YOLOInference = Depends(get_model),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service health status and model information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=model.device,
        model_path=str(model.model_path),
    )


@router.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(..., description="Image file to run inference on"),
    conf: float = 0.25,
    model: YOLOInference = Depends(get_model),
) -> InferResponse:
    """
    Run inference on uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)
        conf: Confidence threshold (0.0-1.0)

    Returns:
        Detection results with bounding boxes
    """
    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Must be image/jpeg or image/png"
        )

    try:
        # Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Run inference
        start_time = time.time()
        detections = model.predict(frame, conf=conf, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Convert to schema
        detection_schemas = [
            DetectionSchema(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                center=det.center,
            )
            for det in detections
        ]

        return InferResponse(
            detections=detection_schemas,
            inference_time_ms=inference_time,
            image_size=(frame.shape[1], frame.shape[0]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/config/rois", response_model=List[CameraROISchema])
async def get_rois(
    roi_manager: ROIManager = Depends(get_roi_manager),
) -> List[CameraROISchema]:
    """
    Get ROI configurations for all cameras.

    Returns:
        List of camera ROI configurations
    """
    rois = []
    for camera_id, camera_roi in roi_manager.camera_rois.items():
        forbidden_zones = [
            {
                "name": zone.name,
                "points": zone.points.tolist(),
            }
            for zone in camera_roi.forbidden_zones
        ]

        rois.append(
            CameraROISchema(
                camera_id=camera_id,
                forbidden_zones=forbidden_zones,
                allowed_classes=list(camera_roi.allowed_classes),
                alert_classes=list(camera_roi.alert_classes),
            )
        )

    return rois


@router.get("/config/cameras", response_model=List[CameraSchema])
async def get_cameras(
    config_loader: ConfigLoader = Depends(get_config_loader),
) -> List[CameraSchema]:
    """
    Get camera configurations.

    Returns:
        List of camera configurations (without RTSP URLs for security)
    """
    cameras_config = config_loader.load_cameras()
    return [
        CameraSchema(
            id=cam.id,
            name=cam.name,
            enabled=cam.enabled,
            fps_target=cam.fps_target,
        )
        for cam in cameras_config.cameras
    ]
