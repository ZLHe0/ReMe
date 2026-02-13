#!/usr/bin/env python3
"""
Qwen3 Embedding vLLM Server (OpenAI-compatible embeddings).
"""

import os
import sys
import time
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Qwen3 Embedding vLLM Server")
parser.add_argument("--model_path", type=str, required=True, help="HF ID or local path")
parser.add_argument("--port", type=int, default=5002, help="Port to run server on")
parser.add_argument("--device", type=str, default="1", help="GPU device IDs")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
parser.add_argument("--pid_file", type=str, default="/tmp/qwen3_embedding.pid", help="PID file")
parser.add_argument("--log_file", type=str, default=None, help="Log file")
parser.add_argument("--initial_wait", type=int, default=120, help="Initial wait time in seconds")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"

try:
    import vllm  # noqa: F401
except ImportError:
    print("Error: vLLM is not installed. Please install vllm.")
    sys.exit(1)

if args.log_file:
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(args.log_file, "w")
else:
    log_file = None

with open(args.pid_file, "w") as f:
    f.write(str(os.getpid()))

def signal_handler(sig, frame):
    if server_process and server_process.poll() is None:
        server_process.terminate()
        server_process.wait(timeout=10)
    if os.path.exists(args.pid_file):
        os.remove(args.pid_file)
    if log_file:
        log_file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_vllm_server():
    cmd = [
        "vllm", "serve", args.model_path,
        "--served-model-name", "qwen3-embedding",
        "--trust-remote-code",
        "--task", "embedding",
        "--gpu-memory-utilization", "0.95",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--uvicorn-log-level", "info",
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file if log_file else subprocess.STDOUT,
    )

def check_server_health(max_retries=60, retry_interval=5):
    import requests
    from requests.exceptions import ConnectionError

    time.sleep(args.initial_wait)
    for _ in range(max_retries):
        try:
            r = requests.get(f"http://localhost:{args.port}/v1/models", timeout=10)
            if r.status_code == 200:
                return True
        except ConnectionError:
            pass
        time.sleep(retry_interval)
    return False

if __name__ == "__main__":
    server_process = start_vllm_server()
    if not check_server_health():
        if server_process and server_process.poll() is None:
            server_process.terminate()
        if os.path.exists(args.pid_file):
            os.remove(args.pid_file)
        if log_file:
            log_file.close()
        sys.exit(1)

    try:
        while server_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    if os.path.exists(args.pid_file):
        os.remove(args.pid_file)
    if log_file:
        log_file.close()
