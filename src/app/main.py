import io
import time
import numpy as np
import onnxruntime as ort
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager

# Config
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45
INPUT_SIZE = 320

model_assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = Path(__file__).resolve().parent
    # Check local dev path vs docker path
    local_path = current_dir.parent.parent / "models" / "production" / "inventory_monitor_quantized.onnx"
    docker_path = current_dir / "models" / "production" / "inventory_monitor_quantized.onnx"

    model_path = local_path if local_path.exists() else docker_path

    if model_path.exists():
        try:
            # Force CPU provider for Fargate stability
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            model_assets["session"] = session
            model_assets["input_name"] = session.get_inputs()[0].name
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    yield
    model_assets.clear()


app = FastAPI(lifespan=lifespan)


def get_processed_detections(predictions, orig_size):
    orig_h, orig_w = orig_size
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    boxes = []
    confidences = []

    for pred in predictions:
        conf = float(pred[4])
        if conf >= CONFIDENCE_THRESHOLD:
            w = int(pred[2] * scale_x)
            h = int(pred[3] * scale_y)
            x = int((pred[0] * scale_x) - w / 2)
            y = int((pred[1] * scale_y) - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "confidence": round(confidences[i], 3),
                "label": "package"
            })
    return results


@app.get("/health")
async def health():
    return {"status": "online", "model_ready": "session" in model_assets}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "session" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.time()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_size = (image.height, image.width)

        # 1. Pre-processing
        t_pre_start = time.time()
        input_img = image.resize((INPUT_SIZE, INPUT_SIZE))
        input_data = np.array(input_img).astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)
        t_pre = (time.time() - t_pre_start) * 1000

        # 2. Inference
        t_inf_start = time.time()
        outputs = model_assets["session"].run(None, {model_assets["input_name"]: input_data})
        t_inf = (time.time() - t_inf_start) * 1000

        # 3. Post-processing (NMS)
        t_post_start = time.time()
        predictions = np.squeeze(outputs[0]).T
        detections = get_processed_detections(predictions, orig_size)
        t_post = (time.time() - t_post_start) * 1000

        # Total processing time
        total_ms = (time.time() - t_start) * 1000

        # Log internal traces to console for debugging (not returned to client)
        print(
            f"ONNX Trace | Pre: {t_pre:.2f}ms | Inf: {t_inf:.2f}ms | Post: {t_post:.2f}ms | Total: {total_ms:.2f}ms")

        return {
            "success": True,
            "package_count": len(detections),
            "inference_time_ms": round(total_ms, 2),
            "detections": detections
        }
    except Exception as e:
        return {"success": False, "error": str(e)}