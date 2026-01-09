import os
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION (Approach A: Global Variables)
# ==========================================
EPOCHS = 50
IMGSZ = 640
BATCH_SIZE = 16
BASE_MODEL = "yolov8s.pt"
PROJECT_NAME = "models"
RUN_NAME = "inventory_monitor_v2"


# ==========================================

def run_training():
    # Hardware acceleration setup
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print(f"[INFO] Training starting on device: {device.upper()}")

    # Path management
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(BASE_DIR, PROJECT_NAME, RUN_NAME, "weights", "last.pt")
    yaml_path = os.path.join(BASE_DIR, "data", "data.yaml")


    # Smart Resume Logic
    if os.path.exists(checkpoint_path):
        print(f"[RESUME] Loading existing checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        resume_training = True
    else:
        print(f"[NEW] No checkpoint found. Initializing with: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume_training = False

    # 1. START TRAINING
    print(f"[TRAIN] Starting training: {EPOCHS} epochs at {IMGSZ}px...")
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        resume=resume_training
    )

    # 2. FINAL EVALUATION
    print(f"\n[TEST] Running final evaluation on Test Set...")
    test_results = model.val(
        data=yaml_path,
        split='test',
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device
    )
    print(f"[INFO] Test Results (mAP50-95): {test_results.box.map}")

    # 3. EXPORT
    print("[EXPORT] Exporting final model to ONNX...")
    model.export(format="onnx")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as error:
        print(f"[ERROR] A critical error occurred: {error}")