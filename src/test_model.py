import os
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
IMGSZ = 320
BATCH_SIZE = 32
PROJECT_FOLDER_NAME = "models"
RUN_NAME = "inventory_monitor"


# ==========================================

def run_final_test():
    # 1. Hardware acceleration setup
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print(f"[INFO] Testing starting on device: {device.upper()}")

    # 2. PATH MANAGEMENT (Relativi alla posizione dello script)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ABSOLUTE_PATH = os.path.join(BASE_DIR, PROJECT_FOLDER_NAME)
    yaml_path = os.path.join(BASE_DIR, "data", "data.yaml")

    best_model_path = os.path.join(MODELS_ABSOLUTE_PATH, RUN_NAME, "weights", "best.pt")

    # 3. Verify model exists
    if not os.path.exists(best_model_path):
        print(f"[ERROR] The model 'best.pt' is not found in: {best_model_path}")
        return

    # 4. LOAD MODEL
    print(f"[INFO] Loading BEST weights from: {best_model_path}")
    model = YOLO(best_model_path)

    # 5. FINAL EVALUATION ON TEST SET
    print(f"\n[TEST] Running evaluation on the 'test' split...")
    results = model.val(
        data=yaml_path,
        split='test',
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        project=MODELS_ABSOLUTE_PATH,
        name=RUN_NAME,
        exist_ok=True
    )

    # 6. PRINT SUMMARY
    print("\n" + "=" * 30)
    print("📊 TEST RESULTS SUMMARY")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("=" * 30)

    # 7. EXPORT TO ONNX
    print("\n[EXPORT] Exporting best model to ONNX for production...")
    model.export(format="onnx")
    print(f"[SUCCESS] Test completed. Results saved in: {MODELS_ABSOLUTE_PATH}/{RUN_NAME}")


if __name__ == "__main__":
    run_final_test()
