"""
Integration Tests for Inference API

Tests FastAPI endpoints with test client.
"""

import pytest
from fastapi.testclient import TestClient


# Note: These tests require a trained model to run
# Mark as integration tests that can be skipped in CI
pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires trained YOLO model")
def test_health_endpoint():
    """Test health check endpoint."""
    from agave_vision.services.inference_api.app import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data


@pytest.mark.skip(reason="Requires trained YOLO model and test image")
def test_infer_endpoint():
    """Test inference endpoint with test image."""
    from agave_vision.services.inference_api.app import create_app
    import io
    from PIL import Image
    import numpy as np

    app = create_app()
    client = TestClient(app)

    # Create a test image
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Send request
    response = client.post(
        "/infer",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
        data={"conf": 0.25},
    )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "inference_time_ms" in data
    assert "image_size" in data


@pytest.mark.skip(reason="Requires configuration files")
def test_get_rois_endpoint():
    """Test get ROIs configuration endpoint."""
    from agave_vision.services.inference_api.app import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/config/rois")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.skip(reason="Requires configuration files")
def test_get_cameras_endpoint():
    """Test get cameras configuration endpoint."""
    from agave_vision.services.inference_api.app import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/config/cameras")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
