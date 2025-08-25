# tiny-knn

A tiny KNN library for finding the top-K nearest neighbors in large embedding spaces, optimized for memory efficiency.

This library works with Torch tensors, NumPy arrays, or file paths to `.pt`/`.npy` arrays. It uses GPU acceleration (via PyTorch) for fast inner-product or cosine similarity. It intelligently batches queries and chunks document matrices to manage memory.

## Features

-   **Memory efficient:** Streams large arrays from disk in manageable chunks.
-   **GPU accelerated:** Leverages PyTorch for fast computation on CUDA-enabled GPUs.
-   **Mixed precision:** Supports `float32`, `float16` (and `bfloat16` for Torch tensors).
-   **Simple I/O:** Accepts Torch tensors, NumPy arrays, or file paths to `.pt`/`.npy`.
-   **CLI + Library:** Use from the command line or directly in Python.

## Installation

```bash
pip install .
```

## Usage

### As a Library

```python
from tiny_knn import exact_search
import torch
import numpy as np

# Torch tensors
Q, D, dim, k = 1000, 1_000_000, 128, 100
queries_t = torch.randn(Q, dim, dtype=torch.float32)
docs_t = torch.randn(D, dim, dtype=torch.float32)
indices_t, scores_t = exact_search(queries_t, docs_t, k, metric="ip")

# NumPy arrays
queries_np = np.random.randn(Q, dim).astype(np.float32)
docs_np = np.random.randn(D, dim).astype(np.float32)
indices_np, scores_np = exact_search(queries_np, docs_np, k, metric="cosine")

# File paths (.pt or .npy)
indices, scores = exact_search("path/to/queries.pt", "path/to/docs.pt", 100, metric="ip")
```

The function returns a tuple `(indices, scores)` where both are the same type family as the inputs:
- Torch input → returns `torch.Tensor` results.
- NumPy input or `.npy` paths → returns `numpy.ndarray` results.

Shapes: `indices.shape == scores.shape == (Q, k)`. `indices` are `int64`; `scores` are `float32`.


### As a Command-Line Tool

`tiny-knn` also provides a command-line interface.

```bash
tiny-knn path/to/queries.(pt|npy) path/to/docs.(pt|npy) --k 100 --metric ip --output-path results.pt
```

Arguments:
- `queries_path`: Path to queries (`.pt` or `.npy`).
- `docs_path`: Path to docs (`.pt` or `.npy`).
- `-k, --k`: Top-K per query to keep (default: 100).
- `--metric`: Similarity metric: `ip` (inner-product) or `cosine`.
- `--output-path`: Optional path to save results (`.pt` or `.npz`).

## API Reference

### `exact_search(arr1, arr2, k, metric)`

```python
def exact_search(
    arr1: np.ndarray | torch.Tensor | str,
    arr2: np.ndarray | torch.Tensor | str,
    k: int,
    metric: str | None = None,
) -> tuple[Indices, Scores]
```

Computes exact top-K nearest neighbors using inner-product (`ip`) or cosine similarity (`cosine`).

- `arr1`, `arr2`: 2D arrays of shape `(N, dim)` provided as Torch tensors, NumPy arrays, or file paths to `.pt`/`.npy`.
- `k`: Number of neighbors to retrieve per query. Must satisfy `1 <= k <= D`.
- `metric`: `'ip'` or `'cosine'`.

Returns `(indices, scores)` where both are either Torch tensors or NumPy arrays, matching the input type. Shapes: `(Q, k)`.

Notes:
- Device and precision are chosen automatically (CUDA if available). Cosine similarity applies L2-normalization internally.
- The implementation batches queries and streams document chunks to control memory usage.

## Development

To install the package in editable mode for development, run:

```bash
pip install -e .
```

## Releases

- GitHub releases: https://github.com/noosphaerenbauer/tiny-knn/releases
- Changelog: see `CHANGELOG.md`
