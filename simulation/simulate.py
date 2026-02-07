import os
import time
import io
import requests
from PIL import Image

# Configuration
API_URL = "http://inventory-monitor-denis.duckdns.org:8000/predict"
IMAGE_DIR = "sample_images"
TARGET_FPS = 1.0
INPUT_SIZE = 320


def run_simulation():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Folder '{IMAGE_DIR}' not found.")
        return

    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images:
        print("No images found to send.")
        return

    print(f"Starting simulation: {len(images)} images at {TARGET_FPS} FPS")
    print(f"Target: {API_URL}\n")

    for filename in images:
        start_time = time.perf_counter()
        path = os.path.join(IMAGE_DIR, filename)

        try:
            with Image.open(path) as img:
                #for png images
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((INPUT_SIZE, INPUT_SIZE))

                buf = io.BytesIO()
                img_resized.save(buf, format="JPEG", quality=85)
                buf.seek(0)

                files = {"file": (filename, buf, "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=10)

            duration = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                print(f"Sent {filename} | Count: {result.get('count')} | {duration:.2f}ms")
            else:
                print(f"Failed {filename} | Status: {response.status_code}")

        except Exception as e:
            print(f"Error sending {filename}: {e}")
            continue

        elapsed = time.perf_counter() - start_time
        sleep_time = (1.0 / TARGET_FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nSimulation complete.")


if __name__ == "__main__":
    time.sleep(3)
    run_simulation()