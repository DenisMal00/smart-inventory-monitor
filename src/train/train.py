import os
import torch
from ultralytics import YOLO

# Training settings
EPOCHS = 100
IMG_SIZE = 320
BATCH_SIZE = 32
MAX_DETECTIONS = 70
MODELS_DIR = "models"
RUN_NAME = "inventory_monitor"


def start_training():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print(f"Training session starting on: {device.upper()}")

    # Setup filesystem paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(project_root, MODELS_DIR)
    data_config = os.path.join(project_root, "data", "data.yaml")

    # Path for resume check
    last_checkpoint = os.path.join(models_path, RUN_NAME, "weights", "last.pt")

    # Resume logic or fresh start
    if os.path.exists(last_checkpoint):
        print(f"Checkpoint found. Resuming training from: {last_checkpoint}")
        model = YOLO(last_checkpoint)
        should_resume = True
    else:
        print("Starting a fresh run for the single-class model.")
        base_model = os.path.join(models_path, "yolov8n.pt")

        if not os.path.exists(base_model):
            print("yolov8n.pt missing in models folder, downloading default...")
            model = YOLO("yolov8n.pt")
        else:
            print(f"Loading clean base weights from: {base_model}")
            model = YOLO(base_model)

        should_resume = False

    # Start the training process
    model.train(
        data=data_config,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        max_det=MAX_DETECTIONS,
        device=device,
        project=models_path,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        resume=should_resume,
        patience=20,
        close_mosaic=10
    )


if __name__ == "__main__":
    start_training()
