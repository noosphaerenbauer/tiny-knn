import os
import time
import torch
from tiny_knn.api import compute_topk

# Map string to torch dtype
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}


def benchmark(q_shape, d_shape, dtype_str, k):
    """
    Generates random data and benchmarks the compute_topk function.
    """
    print("-" * 80)
    print(f"Benchmarking with: Q={q_shape}, D={d_shape}, dtype={dtype_str}, k={k}")

    q_path = "temp_queries.pt"
    d_path = "temp_docs.pt"
    output_path = "temp_results.pt"

    # Generate random data with explicit dtype handling (torch-only)
    if dtype_str == "int8":
        # randint supports negative ranges; use int16 then cast for safety
        q_t = torch.randint(-128, 128, q_shape, dtype=torch.int16).to(torch.int8)
        d_t = torch.randint(-128, 128, d_shape, dtype=torch.int16).to(torch.int8)
    else:
        dtype = DTYPE_MAP.get(dtype_str, torch.float32)
        q_t = torch.rand(q_shape, dtype=torch.float32).to(dtype)
        d_t = torch.rand(d_shape, dtype=torch.float32).to(dtype)

    torch.save(q_t, q_path)
    torch.save(d_t, d_path)

    # Run the benchmark
    start_time = time.time()
    compute_topk(
        q_path=q_path,
        d_path=d_path,
        k=k,
        out_path=output_path,
    )
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")

    # Clean up
    os.remove(q_path)
    os.remove(d_path)
    os.remove(output_path)


def main():
    # Define benchmark configurations
    configs = [
        # Small matrices
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "int8", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "qint8", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "quint4x2", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float8_e4m3fn", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float8_e5m2", "k": 100},
        # Medium matrices
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "float32", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "int8", "k": 100},
        # Large matrices
        {"q_shape": (100000, 768), "d_shape": (1000000, 768), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 768), "d_shape": (1000000, 768), "dtype_str": "float32", "k": 100},
        # Very large matrices
        {"q_shape": (100000, 1024), "d_shape": (1000000, 1024), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 1024), "d_shape": (1000000, 1024), "dtype_str": "float32", "k": 100},
        # Extreme cases
        {"q_shape": (100000, 2048), "d_shape": (1000000, 2048), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 2048), "d_shape": (1000000, 2048), "dtype_str": "float32", "k": 100},
    ]

    for config in configs:
        benchmark(**config)


if __name__ == "__main__":
    main()
