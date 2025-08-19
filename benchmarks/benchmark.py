import os
import time
import torch
from tiny_knn.api import exact_search

# Map string to torch dtype (supported only)
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _gen_tensor(shape, dtype_str):
    dtype = DTYPE_MAP[dtype_str]
    return (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dtype)


def benchmark(q_shape, d_shape, dtype_str, k, normalize):
    metric = "cosine" if normalize else "ip"
    print("-" * 80)
    print(f"Benchmarking with: Q={q_shape}, D={d_shape}, dtype={dtype_str}, k={k}, metric={metric}")

    q_path = f"temp_queries_{os.getpid()}.pt"
    d_path = f"temp_docs_{os.getpid()}.pt"
    output_path = f"temp_results_{os.getpid()}.pt"

    q_t = _gen_tensor(q_shape, dtype_str)
    d_t = _gen_tensor(d_shape, dtype_str)

    torch.save(q_t, q_path)
    torch.save(d_t, d_path)

    start_time = time.time()
    exact_search(
        q_path=q_path,
        d_path=d_path,
        k=k,
        normalize=normalize,
        dtype=dtype_str,
        autotune=True,
        out_path=output_path,
        quiet=True,
    )
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")

    os.remove(q_path)
    os.remove(d_path)
    os.remove(output_path)


def main():
    configs = [
        # Small matrices
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100, "normalize": False},
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "float32", "k": 100, "normalize": True},
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "bfloat16", "k": 100, "normalize": False},
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "bfloat16", "k": 100, "normalize": True},
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100, "normalize": False},
        {"q_shape": (2000, 128), "d_shape": (1000000, 128), "dtype_str": "float16", "k": 100, "normalize": True},
        # Medium matrices
        {"q_shape": (2000, 512), "d_shape": (1000000, 512), "dtype_str": "float32", "k": 100, "normalize": False},
        {"q_shape": (2000, 512), "d_shape": (1000000, 512), "dtype_str": "bfloat16", "k": 100, "normalize": False},
        # Large matrices
        {"q_shape": (2000, 2048), "d_shape": (1000000, 2048), "dtype_str": "float32", "k": 100, "normalize": False},
        {"q_shape": (2000, 2048), "d_shape": (1000000, 2048), "dtype_str": "bfloat16", "k": 100, "normalize": False},
    ]

    for config in configs:
        benchmark(**config)


if __name__ == "__main__":
    main()
