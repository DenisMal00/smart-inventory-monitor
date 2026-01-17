import time
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PT_PATH = os.path.join(BASE_DIR, "models/inventory_monitor/weights/best.pt")
ONNX_FP32_PATH = os.path.join(BASE_DIR, "models/production/inventory_monitor.onnx")
ONNX_INT8_PATH = os.path.join(BASE_DIR, "models/production/inventory_monitor_quantized.onnx")

def benchmark_pt(model_path, input_tensor, iterations=100):
    model = YOLO(model_path)
    # Warmup
    for _ in range(10):
        model.predict(input_tensor, verbose=False)

    start = time.time()
    for _ in range(iterations):
        model.predict(input_tensor, verbose=False)
    return ((time.time() - start) / iterations) * 1000


def benchmark_onnx(model_path, input_tensor, iterations=100):
    # Use CPUExecutionProvider to simulate AWS Fargate environment
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_tensor})

    start = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: input_tensor})
    return ((time.time() - start) / iterations) * 1000


if __name__ == "__main__":
    # Create a dummy input tensor (BCHW: 1, 3, 320, 320)
    dummy_input_np = np.random.rand(1, 3, 320, 320).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    print("Starting Benchmark (Average of 100 iterations)...")

    # 1. PyTorch Benchmark
    lt_pt = benchmark_pt(PT_PATH, dummy_input_torch)
    print(f"PyTorch (.pt):      {lt_pt:.2f} ms")

    # 2. ONNX FP32 Benchmark
    lt_onnx = benchmark_onnx(ONNX_FP32_PATH, dummy_input_np)
    print(f"ONNX (Standard):    {lt_onnx:.2f} ms")

    # 3. ONNX INT8 Benchmark
    lt_int8 = benchmark_onnx(ONNX_INT8_PATH, dummy_input_np)
    print(f"ONNX (Quantized):   {lt_int8:.2f} ms")