# tiny-knn

A tiny KNN library for finding the top-K nearest neighbors in large embedding spaces, optimized for memory efficiency.

This library is designed to work with Torch tensors saved on disk and uses GPU acceleration (via PyTorch) for fast dot-product/cosine similarity computations. It intelligently batches queries and chunks document matrices to manage GPU memory.

## Features

-   **Memory Efficient:** Streams large Torch tensors from disk in manageable chunks.
-   **GPU Accelerated:** Leverages PyTorch for fast computation on CUDA-enabled GPUs.
-   **Mixed Precision:** Supports `float32`, `float16`, and `bfloat16` for computation.
-   **Autotuning:** Can automatically estimate optimal batch and chunk sizes for your hardware.
-   **Command-Line and Library Interface:** Use it as a command-line tool or as a Python library.

## Installation

```bash
pip install .
```

## Usage

### As a Library

```python
from tiny_knn import exact_search

# Paths to your torch tensors (.pt)
queries_path = "path/to/your/queries.pt"
docs_path = "path/to/your/docs.pt"

# Compute the top 100 nearest neighbors
results_path = exact_search(
    q_path=queries_path,
    d_path=docs_path,
    k=100,
    device="cuda",
    normalize=True, # for cosine similarity
)

print(f"Results saved to: {results_path}")
```

The result is a Torch file (`.pt`) containing a dictionary with the following keys:
- `indices`: A torch tensor of shape `(Q, K)` with the indices of the top-K documents for each query.
- `scores`: A torch tensor of shape `(Q, K)` with the similarity scores (float32).
- `queries_path`: The path to the query embeddings file.
- `docs_path`: The path to the document embeddings file.
- `k`: The number of neighbors retrieved.
- `normalize`: Whether cosine similarity was used.
- `batch_size`: The batch size used for the computation.
- `chunk_size`: The chunk size used for the computation.
- `dtype`: The data type used for the computation.
- `device`: The device used for the computation (`cuda` or `cpu`).


### As a Command-Line Tool

`tiny-knn` also provides a command-line interface.

```bash
tiny-knn path/to/queries.pt path/to/docs.pt --k 100 --normalize --output-path results.pt
```

**Arguments:**

-   `queries_path`: Path to queries `.pt` file.
-   `docs_path`: Path to documents `.pt` file.
-   `--output-path`: Path to output `.pt` file.
-   `-k, --k`: Top-K per query to keep (default: 100).
-   `--batch-size`: Query batch size (default: 1024).
-   `--chunk-size`: Document chunk size (default: 65536).
-   `--autotune`: Automatically estimate batch and chunk sizes.
-   `--device`: Device to use (e.g., 'cuda', 'cuda:0', 'cpu', 'mps') (default: 'cuda').
-   `--dtype`: Computation data type ('float32', 'float16', 'bfloat16') (default: 'float32').
-   `--num-workers`: Number of workers for data loading (default: 0).
-   `--normalize`: Normalize vectors to unit length (cosine similarity).
-   `--deterministic`: If True, uses deterministic algorithms.
-   `--verbose`: Enable verbose logging.
-   `--quiet`: Suppress all output except for errors.

## API Reference

### `exact_search(...)`

```python
def exact_search(
    q_path: str,
    d_path: str,
    k: int,
    batch_size: int = 1024,
    chunk_size: int = 65536,
    autotune: bool = False,
    normalize: bool = False,
    deterministic: bool = False,
    device: str = "cuda",
    dtype: str = "float32",
    num_workers: int = 0,
    out_path: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> str:
```

Computes top-K nearest neighbors using exact search.

-   **`q_path`**: Path to the torch-saved tensor of query embeddings (`.pt` or memory-mapped `.npy`).
-   **`d_path`**: Path to the torch-saved tensor of document embeddings (`.pt` or memory-mapped `.npy`).
-   **`k`**: The number of top similarities to retrieve for each query.
-   **`batch_size`**: Query batch size.
-   **`chunk_size`**: Document chunk size.
-   **`autotune`**: If True, automatically estimates batch and chunk sizes.
-   **`normalize`**: If True, normalizes vectors to unit length (for cosine similarity).
-   **`deterministic`**: If True, uses deterministic algorithms.
-   **`device`**: Device to use (e.g., 'cuda', 'cuda:0', 'cpu', 'mps').
-   **`dtype`**: Computation data type ('float32', 'float16', 'bfloat16').
-   **`num_workers`**: Number of workers for data loading (not currently used).
-   **`out_path`**: Optional path to save the results. If None, a path is derived.
-   **`verbose`**: If True, enables verbose logging.
-   **`quiet`**: If True, suppresses all output except for errors.

**Returns:** The path to the saved results file.

## Development

To install the package in editable mode for development, run:

```bash
pip install -e .
```