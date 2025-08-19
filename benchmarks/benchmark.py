import os
import time
import numpy as np
from tiny_knn.api import compute_topk

# Map string to numpy dtype, with a safe fallback for bfloat16
DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": getattr(np, "bfloat16", np.float16),  # fallback to float16 if not available
    "int8": np.int8,
}


def benchmark(q_shape, d_shape, dtype_str, k):
    """
    Generates random data and benchmarks the compute_topk function.
    """
    print("-" * 80)
    print(f"Benchmarking with: Q={q_shape}, D={d_shape}, dtype={dtype_str}, k={k}")

    q_path = "temp_queries.npy"
    d_path = "temp_docs.npy"
    output_path = "temp_results.pkl"

    # Generate random data with explicit dtype handling
    if dtype_str == "int8":
        q_np = np.random.randint(-128, 127, size=q_shape, dtype=np.int8)
        d_np = np.random.randint(-128, 127, size=d_shape, dtype=np.int8)
    else:
        dtype = DTYPE_MAP.get(dtype_str, np.float32)
        q_np = np.random.random_sample(q_shape).astype(dtype, copy=False)
        d_np = np.random.random_sample(d_shape).astype(dtype, copy=False)

    np.save(q_path, q_np)
    np.save(d_path, d_np)

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
        # Medium matrices
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "float32", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "float16", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "int8", "k": 100},
        # Large matrices
        {"q_shape": (100000, 768), "d_shape": (1000000, 768), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 768), "d_shape": (1000000, 768), "dtype_str": "int8", "k": 100},
        # Very large matrices
        {"q_shape": (100000, 1024), "d_shape": (1000000, 1024), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 1024), "d_shape": (1000000, 1024), "dtype_str": "int8", "k": 100},
        # Extreme cases
        {"q_shape": (100000, 2048), "d_shape": (1000000, 2048), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 2048), "d_shape": (1000000, 2048), "dtype_str": "int8", "k": 100},
    ]

    for config in configs:
        benchmark(**config)


if __name__ == "__main__":
    main()
