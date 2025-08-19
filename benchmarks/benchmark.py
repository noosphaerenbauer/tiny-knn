import os
import time
import torch
from tiny_knn.api import exact_search
import numpy as np

# Map string to NumPy dtype (supported only)
NP_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
}


def _write_npy_memmap(path, shape, dtype_str, seed: int | None = 42, max_chunk_mb: int = 128):
    np_dtype = NP_DTYPE_MAP[dtype_str]
    N, dim = shape
    # Create memmap-backed .npy file
    arr = np.lib.format.open_memmap(path, mode='w+', dtype=np_dtype, shape=(N, dim))
    try:
        rng = np.random.default_rng(seed)
        bytes_per_row = dim * np.dtype(np_dtype).itemsize
        chunk_rows = max(1, min(N, (max_chunk_mb * 1024 * 1024) // max(1, bytes_per_row)))
        for i in range(0, N, chunk_rows):
            j = min(i + chunk_rows, N)
            # Generate in fp32 for numerical headroom, then cast on assign
            chunk = rng.random((j - i, dim), dtype=np.float32) * 2.0 - 1.0
            arr[i:j] = chunk.astype(np_dtype, copy=False)
        arr.flush()
    finally:
        del arr
    return path


def benchmark(q_shape, d_shape, dtype_str, k, metric):
    print("-" * 80)
    print(f"Benchmarking with: Q={q_shape}, D={d_shape}, dtype={dtype_str}, k={k}, metric={metric}")

    q_path = "temp_queries.npy"
    d_path = "temp_docs.npy"

    _write_npy_memmap(q_path, q_shape, dtype_str)
    _write_npy_memmap(d_path, d_shape, dtype_str)

    start_time = time.time()
    out_path = exact_search(q_path=q_path, d_path=d_path, k=k, metric=metric, out_path=None)
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")

    # Cleanup
    if os.path.exists(q_path):
        os.remove(q_path)
    if os.path.exists(d_path):
        os.remove(d_path)
    if os.path.exists(out_path):
        os.remove(out_path)


def main():
    configs = [
        # Small matrices
        {"q_shape": (20000, 128), "d_shape": (10000000, 128), "dtype_str": "float32", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 128), "d_shape": (10000000, 128), "dtype_str": "float32", "k": 100, "metric": "cosine"},
        {"q_shape": (20000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100, "metric": "cosine"},
        {"q_shape": (20000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100, "metric": "cosine"},
        {"q_shape": (20000, 128), "d_shape": (10000000, 128), "dtype_str": "float16", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 128), "d_shape": (10000000, 128), "dtype_str": "float16", "k": 100, "metric": "cosine"},
        # Medium matrices
        {"q_shape": (200000, 512), "d_shape": (10000000, 512), "dtype_str": "float32", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 512), "d_shape": (10000000, 512), "dtype_str": "float16", "k": 100, "metric": "ip"},
        # Large matrices
        {"q_shape": (200000, 2048), "d_shape": (10000000, 2048), "dtype_str": "float32", "k": 100, "metric": "ip"},
        {"q_shape": (200000, 2048), "d_shape": (10000000, 2048), "dtype_str": "float16", "k": 100, "metric": "ip"},

    ]

    for config in configs:
        benchmark(**config)


if __name__ == "__main__":
    main()
