# tiny-knn

A tiny KNN library for finding the top-K nearest neighbors in large embedding spaces, optimized for memory efficiency.

This library is designed to work with memory-mapped numpy arrays, allowing you to work with datasets that are much larger than your available RAM. It uses GPU acceleration (via PyTorch) for fast dot-product similarity computations and intelligently batches queries and chunks document matrices to manage GPU memory.

## Features

-   **Memory Efficient:** Uses memory-mapped numpy arrays to handle large embedding files.
-   **GPU Accelerated:** Leverages PyTorch for fast computation on CUDA-enabled GPUs.
-   **Dynamic Chunking:** Automatically estimates the best chunk size for document embeddings to fit into GPU memory.
-   **Command-Line and Library Interface:** Use it as a command-line tool or as a Python library.

## Installation

```bash
pip install .
```

## Usage

### As a Library

You can use `tiny-knn` in your Python code to find the top-K nearest neighbors.

```python
from tiny_knn import compute_topk

# Paths to your memory-mapped numpy arrays
queries_path = "path/to/your/queries.npy"
docs_path = "path/to/your/docs.npy"

# Compute the top 500 nearest neighbors
results_path = compute_topk(
    q_path=queries_path,
    d_path=docs_path,
    k=500,
    dtype_str="float16",  # Use float16 for faster computation
)

print(f"Results saved to: {results_path}")
```

The result is a pickle file containing a dictionary with the following keys:
- `indices`: A numpy array of shape `(Q, K)` with the indices of the top-K documents for each query.
- `scores`: A numpy array of shape `(Q, K)` with the similarity scores.
- `queries_path`: The path to the query embeddings file.
- `docs_path`: The path to the document embeddings file.
- `k`: The number of neighbors retrieved.
- `batch_size`: The batch size used for the computation.
- `dtype`: The data type used for the computation.
- `device`: The device used for the computation (`cuda` or `cpu`).


### As a Command-Line Tool

`tiny-knn` also provides a command-line interface for convenience.

```bash
```bash
tiny-knn path/to/your/queries.npy path/to/your/docs.npy --k 500 --dtype float16
```

**Arguments:**

-   `queries_path`: Path to queries `.npy` file.
-   `docs_path`: Path to documents `.npy` file.
-   `--k`: Top-K per query to keep (default: 2000).
-   `--dtype`: Computation dtype on GPU (`float32`, `float16`, `bfloat16`, `int8`, etc.). Integer types will be upcasted to `float32` for computation. (default: `float32`).
-   `--cpu`: Force CPU (fallback if no CUDA).
-   `--output-path`: Path to output `.pkl` file (optional).
```

**Arguments:**

-   `queries_path`: Path to queries `.npy` file.
-   `docs_path`: Path to documents `.npy` file.
-   `--k`: Top-K per query to keep (default: 2000).
-   `--batch-size`: Number of queries per batch (default: 512).
-   `--dtype`: Computation dtype on GPU (`float32`, `float16`, `bfloat16`, `int8`, etc.). Integer types will be upcasted to `float32` for computation. (default: `float32`).
-   `--cpu`: Force CPU (fallback if no CUDA).
-   `--output-path`: Path to output `.pkl` file (optional).

## Development

To install the package in editable mode for development, run:

```bash
pip install -e .
```
