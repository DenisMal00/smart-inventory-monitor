from ultralytics import YOLO
import torch
import os

def run_training(epochs=1, imgsz=320):
    # Setup hardware
    device = "cpu" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = YOLO('yolov8n.pt')
    yaml_path = os.path.abspath("data/data.yaml")

    # Training
    print(f"Training for {epochs} epochs...")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=device,
        project="models",
        name="inventory_monitor",
        exist_ok=True,
        plots=True
    )

    # Export
    print("Training finished. Exporting to ONNX...")
    model.export(format="onnx")

if __name__ == "__main__":
    run_training(epochs=50, imgsz=320)