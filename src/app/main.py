import io
import os

import numpy as np
import onnxruntime as ort
import cv2
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import BaseModel
import requests

DUCKDNS_TOKEN = os.getenv("DUCKDNS_TOKEN")
DUCKDNS_DOMAIN = os.getenv("DUCKDNS_DOMAIN")

# --- Configuration & Global State ---
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45

inventory_state = {
    "current_count": 0,
    "critical_threshold": 2,
    "full_capacity": 6,
    "last_check": "Never",
    "status": "WAITING",
    "message": "Waiting for first detection...",
    "history": []
}

model_assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized model loading for single-core Fargate environment."""

    try:
        url = f"https://www.duckdns.org/update?domains={DUCKDNS_DOMAIN}&token={DUCKDNS_TOKEN}"
        requests.get(url, timeout=10)
        print(f"DuckDNS updated for domain: {DUCKDNS_DOMAIN}")
    except Exception as e:
        print(f"DuckDNS update failed: {e}")

    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "models" / "production" / "inventory_monitor_quantized.onnx"

    if model_path.exists():
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

            model_assets["session"] = session
            model_assets["input_name"] = session.get_inputs()[0].name
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    yield
    model_assets.clear()


app = FastAPI(lifespan=lifespan)


# --- Internal Logic ---

def get_processed_detections(predictions, orig_size):
    """Core logic to filter boxes using NMS."""
    orig_h, orig_w = orig_size
    scale_x, scale_y = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE

    boxes, confidences = [], []
    for pred in predictions:
        conf = float(pred[4])
        if conf >= CONFIDENCE_THRESHOLD:
            w, h = int(pred[2] * scale_x), int(pred[3] * scale_y)
            x, y = int((pred[0] * scale_x) - w / 2), int((pred[1] * scale_y) - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return indices, boxes, confidences


def update_inventory_status(count):
    """Updates global state and history buffer."""
    inventory_state["current_count"] = count
    inventory_state["last_check"] = datetime.now().strftime("%H:%M:%S")

    if count >= inventory_state["full_capacity"]:
        inventory_state["status"] = "FULL"
        inventory_state["message"] = "Optimal Level"
    elif count > inventory_state["critical_threshold"]:
        inventory_state["status"] = "WARNING"
        inventory_state["message"] = "Stock Low"
    else:
        inventory_state["status"] = "CRITICAL"
        inventory_state["message"] = "Emergency Restock"

    log_entry = {"timestamp": inventory_state["last_check"], "count": count, "status": inventory_state["status"]}
    inventory_state["history"].insert(0, log_entry)
    inventory_state["history"] = inventory_state["history"][:20]


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/inspector", response_class=HTMLResponse)
async def inspector_page():
    return (Path(__file__).parent / "inspector.html").read_text()


@app.get("/status")
async def get_status():
    return inventory_state


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "session" not in model_assets: raise HTTPException(status_code=503)
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_img = image.resize((INPUT_SIZE, INPUT_SIZE), resample=Image.BILINEAR)
        input_data = np.array(input_img).astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)

        outputs = model_assets["session"].run(None, {model_assets["input_name"]: input_data})
        indices, _, _ = get_processed_detections(np.squeeze(outputs[0]).T, (image.height, image.width))
        count = len(indices.flatten()) if len(indices) > 0 else 0
        update_inventory_status(count)
        return {"success": True, "count": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/verify-image")
async def verify_image(file: UploadFile = File(...)):
    """Inspector endpoint fixed with RGB conversion for accuracy."""
    if "session" not in model_assets:
        raise HTTPException(status_code=503)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig_h, orig_w = img.shape[:2]

        # correction of the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        input_data = input_img.astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)

        outputs = model_assets["session"].run(None, {model_assets["input_name"]: input_data})

        #NMS
        indices, boxes, confidences = get_processed_detections(np.squeeze(outputs[0]).T, (orig_h, orig_w))

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (138, 43, 226), 3)

        _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ThresholdSettings(BaseModel):
    min: int
    max: int


@app.post("/update-settings")
async def update_settings(settings: ThresholdSettings):
    try:
        inventory_state["critical_threshold"] = settings.min
        inventory_state["full_capacity"] = settings.max

        if inventory_state["last_check"] != "Never":
            update_inventory_status(inventory_state["current_count"])
        return {"success": True, "updated": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

