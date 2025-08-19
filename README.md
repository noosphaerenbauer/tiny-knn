# tiny-knn

A tiny KNN library for finding the top-K nearest neighbors in large embedding spaces, optimized for memory efficiency.

This library is designed to work with Torch tensors saved on disk and uses GPU acceleration (via PyTorch) for fast dot-product similarity computations. It intelligently batches queries and chunks document matrices to manage GPU memory.

## Features

-   **Memory Efficient:** Streams large Torch tensors from disk in manageable chunks.
-   **GPU Accelerated:** Leverages PyTorch for fast computation on CUDA-enabled GPUs.
-   **Dynamic Chunking:** Automatically estimates the best batch size and document chunk sizes to fit into GPU memory.
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

# Paths to your torch tensors (.pt)
queries_path = "path/to/your/queries.pt"
docs_path = "path/to/your/docs.pt"

# Compute the top 500 nearest neighbors
results_path = compute_topk(
    q_path=queries_path,
    d_path=docs_path,
    k=500,
)

print(f"Results saved to: {results_path}")
```

The result is a Torch file (.pt) containing a dictionary with the following keys:
- `indices`: A torch tensor of shape `(Q, K)` with the indices of the top-K documents for each query.
- `scores`: A torch tensor of shape `(Q, K)` with the similarity scores (float32).
- `queries_path`: The path to the query embeddings file (.pt).
- `docs_path`: The path to the document embeddings file (.pt).
- `k`: The number of neighbors retrieved.
- `batch_size`: The batch size used for the computation.
- `dtype`: The data type used for the computation.
- `device`: The device used for the computation (`cuda` or `cpu`).


### As a Command-Line Tool

`tiny-knn` also provides a command-line interface for convenience.

```bash
tiny-knn path/to/your/queries.pt path/to/your/docs.pt --k 500
```

**Arguments:**

-   `queries_path`: Path to queries `.pt` file.
-   `docs_path`: Path to documents `.pt` file.
-   `--k`: Top-K per query to keep (default: 2000).
-   `--cpu`: Force CPU (fallback if no CUDA).
-   `--output-path`: Path to output `.pt` file (optional).

## Development

To install the package in editable mode for development, run:

```bash
pip install -e .
```
