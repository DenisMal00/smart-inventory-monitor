import os
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
ADDITIONAL_EPOCHS = 50
IMGSZ = 320
BATCH_SIZE = 16
PROJECT_FOLDER_NAME = "models"
RUN_NAME = "inventory_monitor"


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

    # =========================================================
    # PATH MANAGEMENT
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ABSOLUTE_PATH = os.path.join(BASE_DIR, PROJECT_FOLDER_NAME)
    yaml_path = os.path.join(BASE_DIR, "data", "data.yaml")

    checkpoint_path = os.path.join(MODELS_ABSOLUTE_PATH, "inventory_monitor", "weights", "last.pt")
    # =========================================================

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    print(f"[INFO] Loading weights from: {checkpoint_path}")
    print(f"[INFO] Starting NEW training run with pretrained weights")
    model = YOLO(checkpoint_path)

    # START NEW TRAINING (not resume)
    print(f"[TRAIN] Training for {ADDITIONAL_EPOCHS} additional epochs...")

    model.train(
        data=yaml_path,
        epochs=ADDITIONAL_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        project=MODELS_ABSOLUTE_PATH,
        name=RUN_NAME,
        close_mosaic=50,
        exist_ok=True,
        plots=True,
        save=True,
        resume=True,
        patience=15
    )

if __name__ == "__main__":
    try:
        run_training()
    except Exception as error:
        print(f"[ERROR] A critical error occurred: {error}")