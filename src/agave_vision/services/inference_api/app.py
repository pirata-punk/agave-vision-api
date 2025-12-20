"""
Inference API Application

FastAPI application factory with lifecycle management.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agave_vision.config.loader import ConfigLoader
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager
from agave_vision.utils.logging import setup_logging, get_logger

from . import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads model and configurations on startup, cleans up on shutdown.
    """
    logger = get_logger(__name__)

    # Load configurations
    logger.info("Loading configurations...")
    config_loader = ConfigLoader(Path("configs"))
    services_config = config_loader.load_services()

    # Load YOLO model
    logger.info(f"Loading YOLO model from {services_config.inference.model_path}...")
    model = YOLOInference(
        model_path=services_config.inference.model_path,
        conf=services_config.inference.confidence,
        iou=services_config.inference.iou_threshold,
        device=services_config.inference.device,
        imgsz=services_config.inference.image_size,
    )

    # Warmup model
    logger.info(f"Warming up model ({services_config.inference.warmup_iterations} iterations)...")
    model.warmup(iterations=services_config.inference.warmup_iterations)
    logger.info("Model warmup complete")

    # Load ROI manager
    logger.info("Loading ROI configurations...")
    roi_manager = ROIManager(Path("configs/rois.yaml"))
    logger.info(f"Loaded ROIs for {len(roi_manager.camera_rois)} cameras")

    # Store in app state
    app.state.model = model
    app.state.roi_manager = roi_manager
    app.state.config_loader = config_loader

    logger.info("Inference API ready")

    yield  # Application is running

    # Cleanup (on shutdown)
    logger.info("Shutting down Inference API...")


def create_app() -> FastAPI:
    """
    Create FastAPI application instance.

    Returns:
        Configured FastAPI application
    """
    # Setup logging
    logger = setup_logging("inference-api", level="INFO", format="json")

    # Create FastAPI app
    app = FastAPI(
        title="Agave Vision Inference API",
        description="YOLOv8 object detection API for industrial cameras",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(routes.router, tags=["inference"])

    logger.info("Inference API application created")

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agave_vision.services.inference_api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
