import os
import time
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def _derive_output_path(queries_path: str, docs_path: str, k: int) -> str:
    """Derives a default output path from the input paths."""
    qbase = os.path.splitext(os.path.basename(queries_path))[0]
    dbase = os.path.splitext(os.path.basename(docs_path))[0]
    return f"top{k}_{qbase}_x_{dbase}.pt"


def _get_free_cuda_mem() -> Tuple[int, int]:
    """Returns the free and total CUDA memory in bytes."""
    try:
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return 0, 0


def _estimate_batch_size(Q: int, dim: int, emb_bytes: int, safety_frac: float = 0.5, max_batch_size: int = 524288) -> int:
    """
    Estimate a safe batch size to fit in GPU memory.
    This is a heuristic that allocates a fraction of memory for the query batch.
    """
    free_mem, _ = _get_free_cuda_mem()
    if free_mem <= 0:
        # CPU or unknown: pick a conservative default
        return min(Q, max_batch_size)

    usable = int(free_mem * safety_frac)

    # Allocate ~20% of usable memory to the query batch tensor
    query_mem_limit = int(usable * 0.2)

    # Calculate batch size based on memory for a single query vector
    bs = query_mem_limit // (dim * emb_bytes + 1)

    # Clamp the batch size to reasonable limits
    return max(1, min(bs, Q, max_batch_size))


def _estimate_doc_chunk_rows(D: int, dim: int, batch_size: int, emb_bytes: int, score_bytes: int, safety_frac: float = 0.5, max_chunk_size: int = 524288) -> int:
    """
    Estimate a safe number of document rows to load on GPU per chunk so that:
      - result matrix (batch_size x rows) fits comfortably
      - doc chunk tensor (rows x dim) fits comfortably
    Uses a fraction of currently free memory for safety.
    """
    free_mem, _ = _get_free_cuda_mem()
    if free_mem <= 0:
        # CPU or unknown: pick a conservative default
        return max(1, min(D, max_chunk_size))

    usable = int(free_mem * safety_frac)
    if usable <= 0:
        return max(1, min(D, max_chunk_size))

    # Memory for one row of docs and one row of scores
    mem_per_row_docs = dim * emb_bytes
    mem_per_row_scores = batch_size * score_bytes

    # Total memory per row is the sum of these
    total_mem_per_row = mem_per_row_docs + mem_per_row_scores

    # Calculate how many rows can fit
    rows = usable // (total_mem_per_row + 1)

    # Clip to dataset size and a reasonable maximum chunk size
    return max(1, min(D, max_chunk_size, rows))


def _to_device_chunk(t_cpu: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Moves a CPU torch tensor chunk to the target device with proper dtype, using pinned memory on CUDA."""
    if device.type == "cuda":
        t_cpu = t_cpu.contiguous().pin_memory()
        return t_cpu.to(device=device, dtype=dtype, non_blocking=True)
    else:
        return t_cpu.to(device=device, dtype=dtype)


def _load_tensor(path: str) -> torch.Tensor:
    """Loads a tensor from a .pt or .npy file."""
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    elif path.endswith(".npy"):
        return torch.from_numpy(np.load(path, mmap_mode="r"))
    else:
        raise ValueError(f"Unsupported file format: {path}. Please use .pt or .npy.")


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
    """
    Computes top-K nearest neighbors using exact search.

    Args:
        q_path: Path to the torch-saved tensor of query embeddings (.pt or .npy).
        d_path: Path to the torch-saved tensor of document embeddings (.pt or .npy).
        k: The number of top similarities to retrieve for each query.
        batch_size: Query batch size.
        chunk_size: Document chunk size.
        autotune: If True, automatically estimates batch and chunk sizes.
        normalize: If True, normalizes vectors to unit length (cosine similarity).
        deterministic: If True, uses deterministic algorithms.
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cpu', 'mps').
        dtype: Computation data type ('float32', 'float16', 'bfloat16').
        num_workers: Number of workers for data loading (not currently used).
        out_path: Optional path to save the results. If None, a path is derived.
        verbose: If True, enables verbose logging.
        quiet: If True, suppresses all output except for errors.

    Returns:
        The path to the saved results file.
    """
    start_time = time.time()

    # Setup logging
    log = lambda *args, **kwargs: None
    if not quiet:
        log = print
    if verbose:
        log = lambda *args, **kwargs: print(*args, **kwargs, flush=True)

    # --- Determinism ---
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # --- Device and Dtype Setup ---
    torch_device = torch.device(device)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    # --- Data Loading ---
    log(f"Loading data from {q_path} and {d_path}...")
    q_cpu = _load_tensor(q_path)
    d_cpu = _load_tensor(d_path)

    # --- Input Validation ---
    if q_cpu.dim() != 2 or d_cpu.dim() != 2:
        raise ValueError("Embeddings must be 2D tensors of shape (N, dim)")
    if q_cpu.shape[1] != d_cpu.shape[1]:
        raise ValueError(f"Dim mismatch: queries dim={q_cpu.shape[1]} vs docs dim={d_cpu.shape[1]}")
    if k > d_cpu.shape[0]:
        raise ValueError(f"k ({k}) cannot be larger than the number of documents ({d_cpu.shape[0]})")
    for t, name in [(q_cpu, "queries"), (d_cpu, "docs")]:
        if not isinstance(t, torch.Tensor):
             # This can happen with memmapped numpy arrays
            if np.isnan(t).any() or np.isinf(t).any():
                raise ValueError(f"NaNs or Infs found in {name} tensor.")
        elif torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError(f"NaNs or Infs found in {name} tensor.")

    Q, dim = q_cpu.shape
    D, _ = d_cpu.shape

    # --- Autotuning ---
    if autotune:
        log("Autotuning batch and chunk sizes...")
        emb_bytes = torch.tensor([], dtype=torch_dtype).element_size()
        score_bytes = 4  # fp32 for scores
        batch_size = _estimate_batch_size(Q, dim, emb_bytes)
        chunk_size = _estimate_doc_chunk_rows(D, dim, batch_size, emb_bytes, score_bytes)

    # --- Backend Optimizations ---
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # --- Output Allocation ---
    topk_scores = torch.empty((Q, k), dtype=torch.float32)
    topk_indices = torch.empty((Q, k), dtype=torch.int64)

    log(f"Device={torch_device.type}, queries={Q}x{dim}, docs={D}x{dim}, k={k}, dtype={dtype}, normalize={normalize}, deterministic={deterministic}")
    log(f"Batch size={batch_size}, Chunk size={chunk_size}")

    # --- Main Search Loop ---
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(torch_device.type == "cuda" and torch_dtype != torch.float32)):
        stream = torch.cuda.Stream() if torch_device.type == "cuda" else None
        
        pbar = tqdm(total=Q, desc="Queries", disable=quiet)
        for qs in range(0, Q, batch_size):
            qe = min(qs + batch_size, Q)

            qb = _to_device_chunk(q_cpu[qs:qe], torch_device, torch_dtype)
            if normalize:
                qb = F.normalize(qb, p=2, dim=1)

            batch_scores, batch_indices = None, None

            for ds in range(0, D, chunk_size):
                de = min(ds + chunk_size, D)

                db = _to_device_chunk(d_cpu[ds:de], torch_device, torch_dtype)
                if normalize:
                    db = F.normalize(db, p=2, dim=1)

                scores = torch.matmul(qb, db.t())

                chunk_k = min(k, de - ds)
                vals, idx = torch.topk(scores.float(), k=chunk_k, dim=1, largest=True)
                idx += ds  # Adjust to global indices

                if batch_scores is None:
                    batch_scores, batch_indices = vals, idx
                else:
                    combined_scores = torch.cat([batch_scores, vals], dim=1)
                    combined_indices = torch.cat([batch_indices, idx], dim=1)
                    batch_scores, pick = torch.topk(combined_scores, k=k, dim=1, largest=True)
                    batch_indices = torch.gather(combined_indices, 1, pick)
            
            topk_scores[qs:qe] = batch_scores.cpu()
            topk_indices[qs:qe] = batch_indices.cpu()
            pbar.update(qe - qs)
        pbar.close()

    elapsed = time.time() - start_time
    log(f"Done in {elapsed:.2f}s")

    # --- Persistence ---
    if out_path is None:
        out_path = _derive_output_path(q_path, d_path, k)

    out_obj = {
        "indices": topk_indices,
        "scores": topk_scores,
        "queries_path": q_path,
        "docs_path": d_path,
        "k": k,
        "normalize": normalize,
        "deterministic": deterministic,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "dtype": dtype,
        "device": device,
    }
    torch.save(out_obj, out_path)

    log(f"Saved: {out_path}")
    return out_path
