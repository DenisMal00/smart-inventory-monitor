import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def run_test(image_source, weights='models/inventory_monitor/weights/best.pt', conf=0.25):
    """
    Run inference on a specific source using the trained model weights.
    """
    if not Path(weights).exists():
        print(f"Error: Weights not found at {weights}")
        sys.exit(1)

    # Load model
    model = YOLO(weights)

    # Inference
    results = model.predict(
        source=image_source,
        conf=conf,
        save=True,
        imgsz=320
    )

    # Display loop
    for r in results:
        annotated_frame = r.plot()
        
        # Display window
        window_name = f"Inference - {Path(image_source).name}"
        cv2.imshow(window_name, annotated_frame)
        
        print(f"Results saved to: {r.save_dir}")
        print("Press any key to close...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inventory Monitor - Test Script")
    parser.add_argument("--source", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    run_test(args.source, conf=args.conf)