import os
import time
import requests
import statistics

API_URL = "http://localhost:8000/predict"
IMAGE_FOLDER = "test"
CYCLES = 10
WARMUP_RUNS = 10


def run_benchmark():
    # 1. Setup image list
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"Error: No images found in '{IMAGE_FOLDER}'.")
        return

    total_requests = len(images) * CYCLES
    print(f"Starting Benchmark: {len(images)} images x {CYCLES} cycles = {total_requests} total requests")
    print(f"Warmup: First {WARMUP_RUNS} requests will be discarded from stats.")
    print("-" * 60)

    rtt_times = []  # Total round-trip time (Client side)
    server_times = []  # Internal inference time (Server side)
    errors = 0

    # 2. Main Execution Loop
    for i in range(total_requests):
        img_name = images[i % len(images)]
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            with open(img_path, "rb") as f:
                t_start = time.perf_counter()
                response = requests.post(API_URL, files={"file": f})
                t_end = time.perf_counter()

            if response.status_code == 200:
                if i >= WARMUP_RUNS:
                    duration_ms = (t_end - t_start) * 1000
                    data = response.json()

                    server_ms = data.get("inference_time_ms", 0)

                    rtt_times.append(duration_ms)
                    server_times.append(server_ms)

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{total_requests} requests...")
            else:
                errors += 1
                print(f"Request failed: {response.status_code}")

        except Exception as e:
            errors += 1
            print(f"Connection error: {e}")

    # 3. Final Report
    if rtt_times:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Valid Samples: {len(rtt_times)} (excluding warmup)")
        print(f"Errors:        {errors}")
        print("-" * 30)

        avg_rtt = statistics.mean(rtt_times)
        avg_server = statistics.mean(server_times)

        # Calculate 95th percentile (P95)
        p95_rtt = statistics.quantiles(rtt_times, n=20)[18]

        print(f" Avg Latency (RTT):    {avg_rtt:.2f} ms")
        print(f" Avg Server Inference: {avg_server:.2f} ms")
        print(f" Avg Network Overhead: {(avg_rtt - avg_server):.2f} ms")
        print("-" * 30)
        print(f" Min Latency: {min(rtt_times):.2f} ms")
        print(f" Max Latency: {max(rtt_times):.2f} ms")
        print(f" P95 Latency: {p95_rtt:.2f} ms")
        print("=" * 60)
    else:
        print("No successful data collected.")


if __name__ == "__main__":
    run_benchmark()



'''
Onnx base:
============================================================
BENCHMARK RESULTS
============================================================
Valid Samples: 278 (excluding warmup)
Errors:        0
------------------------------
 Avg Latency (RTT):    831.37 ms
 Avg Server Inference: 763.78 ms
 Avg Network Overhead: 67.59 ms
------------------------------
 Min Latency: 492.23 ms
 Max Latency: 1403.11 ms
 P95 Latency: 1105.48 ms
============================================================


Onnx int8
============================================================
BENCHMARK RESULTS
============================================================
Valid Samples: 950 (excluding warmup)
Errors:        0
------------------------------
 Avg Latency (RTT):    464.39 ms
 Avg Server Inference: 405.47 ms
 Avg Network Overhead: 58.93 ms
------------------------------
 Min Latency: 114.44 ms
 Max Latency: 891.77 ms
 P95 Latency: 698.15 ms
============================================================

Torch
============================================================
BENCHMARK RESULTS
============================================================
Valid Samples: 950 (excluding warmup)
Errors:        0
------------------------------
 Avg Latency (RTT):    462.92 ms
 Avg Server Inference: 439.98 ms
 Avg Network Overhead: 22.94 ms
------------------------------
 Min Latency: 190.13 ms
 Max Latency: 890.21 ms
 P95 Latency: 705.33 ms
============================================================

'''