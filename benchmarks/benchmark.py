import os
import time
import torch
from tiny_knn.api import compute_topk

# Map string to torch dtype (supported only)
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _gen_tensor(shape, dtype_str):
    dtype = DTYPE_MAP[dtype_str]
    return (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dtype)


def benchmark(q_shape, d_shape, dtype_str, k):
    print("-" * 80)
    print(f"Benchmarking with: Q={q_shape}, D={d_shape}, dtype={dtype_str}, k={k}")

    q_path = "temp_queries.pt"
    d_path = "temp_docs.pt"
    output_path = "temp_results.pt"

    q_t = _gen_tensor(q_shape, dtype_str)
    d_t = _gen_tensor(d_shape, dtype_str)

    torch.save(q_t, q_path)
    torch.save(d_t, d_path)

    start_time = time.time()
    compute_topk(q_path=q_path, d_path=d_path, k=k, out_path=output_path)
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")

    os.remove(q_path)
    os.remove(d_path)
    os.remove(output_path)


def main():
    configs = [
        # Small matrices
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "bfloat16", "k": 100},
        {"q_shape": (100000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100},
        # Medium matrices
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "float32", "k": 100},
        {"q_shape": (100000, 256), "d_shape": (1000000, 256), "dtype_str": "bfloat16", "k": 100},
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
