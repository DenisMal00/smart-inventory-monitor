import os
import time
import io
import requests
import statistics
from PIL import Image

# Configuration
api_url = "http://inventory-monitor-denis.duckdns.org:8000/predict"
image_dir = "test"
warmup_runs = 10
test_runs = 50
target_fps = 2.0
input_size = 320  # Matches model input


def run():
    if not os.path.exists(image_dir):
        print(f"Error: folder '{image_dir}' not found.")
        return

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("No test images found.")
        return

    print(f"Starting optimized test: {warmup_runs} warmup + {test_runs} runs @ {target_fps} FPS")
    print(f"Client-side resizing enabled: {input_size}x{input_size}")
    print("-" * 50)

    latencies = []
    errors = 0
    total = warmup_runs + test_runs

    for i in range(total):
        loop_start = time.perf_counter()

        img_name = images[i % len(images)]
        path = os.path.join(image_dir, img_name)
        is_warmup = i < warmup_runs

        try:
            # Resize image in memory before sending
            with Image.open(path) as img:
                img_resized = img.resize((input_size, input_size))

                # Save to buffer as JPEG
                buf = io.BytesIO()
                img_resized.save(buf, format="JPEG", quality=85)
                buf.seek(0)

                # Send the optimized buffer
                resp = requests.post(api_url, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=10)

            duration = (time.perf_counter() - loop_start) * 1000

            if resp.status_code == 200:
                if not is_warmup:
                    latencies.append(duration)

                phase = "WARMUP" if is_warmup else "RUN"
                print(f"[{phase}] {i + 1}/{total} | {img_name} | {duration:.2f}ms")
            else:
                errors += 1
                print(f"Server error {resp.status_code} on {img_name}")

        except Exception as e:
            errors += 1
            print(f"Request failed: {e}")

        # Maintain constant 1 FPS
        elapsed = time.perf_counter() - loop_start
        wait_time = (1.0 / target_fps) - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

    if latencies:
        print("\n" + "=" * 50)
        print("Final Stats (Optimized Upload)")
        print("=" * 50)
        print(f"Samples: {len(latencies)}")
        print(f"Errors:  {errors}")
        print("-" * 25)

        avg_rtt = statistics.mean(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]

        print(f"Avg Latency: {avg_rtt:.2f} ms")
        print(f"P95 Latency: {p95:.2f} ms")
        print(f"Min: {min(latencies):.2f} ms | Max: {max(latencies):.2f} ms")

        # Verdict
        if avg_rtt < 1000:
            print("\nResult: System stable and highly responsive.")
        else:
            print("\nResult: Potential bottleneck even with optimization.")
        print("=" * 50)


if __name__ == "__main__":
    run()