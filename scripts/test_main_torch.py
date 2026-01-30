import io
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager
from ultralytics import YOLO

# Dictionary to store the model in the app context
model_assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = Path(__file__).resolve().parent

    # Path configuration for both local and Docker environments
    local_path = current_dir.parent.parent / "models" / "production" / "inventory_monitor.pt"
    docker_path = current_dir / "models" / "production" / "inventory_monitor.pt"

    model_path = local_path if local_path.exists() else docker_path

    if model_path.exists():
        try:
            # Load the PyTorch model
            model_assets["model"] = YOLO(str(model_path))
            print(f"PyTorch model successfully loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading the PyTorch model: {e}")
    else:
        print(f"⚠️ Model file not found at: {model_path}")

    yield
    # Clean up on shutdown
    model_assets.clear()


app = FastAPI(title="Inventory Monitor - PyTorch Engine", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "online", "engine": "pytorch", "model_loaded": "model" in model_assets}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not initialized")

    t_start = time.time()

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Inference + Internal Processing
        # Ultralytics wraps pre, inf, and post-NMS in one call
        t_inf_start = time.time()
        results = model_assets["model"].predict(
            source=image, conf=0.30, iou=0.45, imgsz=320, verbose=False
        )[0]
        t_inf = (time.time() - t_inf_start) * 1000

        # 2. JSON Formatting (Our overhead)
        t_format_start = time.time()
        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords
            detections.append({
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": round(float(box.conf), 3),
                "label": "package"
            })
        t_format = (time.time() - t_format_start) * 1000

        total_ms = (time.time() - t_start) * 1000

        print(
            f"Torch Trace | Model Call (Pre+Inf+Post): {t_inf:.2f}ms | JSON Format: {t_format:.2f}ms | Total: {total_ms:.2f}ms")

        return {
            "success": True,
            "package_count": len(detections),
            "inference_time_ms": round(total_ms, 2),
            "detections": detections
        }
    except Exception as e:
        return {"success": False, "error": str(e)}