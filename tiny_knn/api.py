import os
import pickle
import time
from typing import Tuple

import numpy as np
import torch


def _torch_dtype(dtype_str: str) -> torch.dtype:
    """Converts a string to a torch.dtype."""
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def _derive_output_path(queries_path: str, docs_path: str, k: int) -> str:
    """Derives a default output path from the input paths."""
    qbase = os.path.splitext(os.path.basename(queries_path))[0]
    dbase = os.path.splitext(os.path.basename(docs_path))[0]
    return f"top{k}_{qbase}_x_{dbase}.pkl"


def _get_free_cuda_mem() -> Tuple[int, int]:
    """Returns the free and total CUDA memory in bytes."""
    try:
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return 0, 0


def _estimate_batch_size(Q: int, dim: int, emb_bytes: int, safety_frac: float = 0.5, max_batch_size: int = 262144) -> int:
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




def _estimate_doc_chunk_rows(D: int, dim: int, batch_size: int, emb_bytes: int, score_bytes: int, safety_frac: float = 0.5, max_chunk_size: int = 262144) -> int:
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


def _to_device_chunk(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Moves a numpy array to a torch device."""
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        t = t.pin_memory()
    return t.to(device=device, dtype=dtype, non_blocking=True)


def compute_topk(q_path: str, d_path: str, k: int, force_cpu: bool = False, out_path: str | None = None) -> str:
    """
    Computes top-K similarities (dot-product) between query and document embeddings.

    Args:
        q_path: Path to the memory-mapped numpy array of query embeddings.
        d_path: Path to the memory-mapped numpy array of document embeddings.
        k: The number of top similarities to retrieve for each query.
        force_cpu: If True, forces the computation to run on the CPU.
        out_path: Optional path to save the results. If None, a path is derived from the input paths.

    Returns:
        The path to the saved results file.
    """
    start_time = time.time()

    # Load with memory mapping to reduce RAM pressure
    q_np = np.load(q_path, mmap_mode="r", allow_pickle=True)
    d_np = np.load(d_path, mmap_mode="r")

    if q_np.ndim != 2 or d_np.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays of shape (N, dim)")
    if q_np.shape[1] != d_np.shape[1]:
        raise ValueError(f"Dim mismatch: queries dim={q_np.shape[1]} vs docs dim={d_np.shape[1]}")
    if q_np.dtype != d_np.dtype:
        raise ValueError(f"Dtype mismatch: queries dtype={q_np.dtype} vs docs dtype={d_np.dtype}")

    Q, dim = q_np.shape
    D, _ = d_np.shape

    device = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")

    # Backend/precision optimizations
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        # Optionally control CPU threading via env var
        try:
            cpu_threads = int(os.environ.get("TINYKNN_CPU_THREADS", "0"))
            if cpu_threads > 0:
                torch.set_num_threads(cpu_threads)
        except Exception:
            pass

    tdtype = _torch_dtype(str(q_np.dtype))
    emb_bytes = torch.tensor([], dtype=tdtype).element_size()
    score_bytes = 4  # keep scores in fp32 for topk stability

    # Dynamically estimate batch size (once upfront)
    batch_size = _estimate_batch_size(Q, dim, emb_bytes)

    # Estimate doc chunk rows once based on the chosen batch size
    doc_rows = _estimate_doc_chunk_rows(D, dim, batch_size, emb_bytes, score_bytes)

    # Cache behavior configuration
    empty_cache_mode = os.environ.get("TINYKNN_EMPTY_CACHE", "batch")  # "never"|"chunk"|"batch"

    # Allocate outputs on CPU (final result)
    topk_scores = np.empty((Q, k), dtype=np.float32)
    topk_indices = np.empty((Q, k), dtype=np.int64)

    print(
        f"Device={device.type}, queries={Q}x{dim}, docs={D}x{dim}, batch_size={batch_size}, doc_rows={doc_rows}, k={k}, dtype={q_np.dtype}")

    with torch.no_grad():
        # Use a separate stream for prefetching data to overlap transfers and compute
        stream = torch.cuda.Stream() if device.type == "cuda" else None

        for qs in range(0, Q, batch_size):
            qe = min(qs + batch_size, Q)
            cur_bs = qe - qs

            # Move query batch to device
            qb = _to_device_chunk(np.array(q_np[qs:qe]), device, tdtype)

            # Running top-k for this batch
            prev_scores = None  # (bs, <=k)
            prev_indices = None  # (bs, <=k)

            # --- Dataloader-style prefetching loop ---
            next_db = None
            if stream:  # Prefetch first chunk
                with torch.cuda.stream(stream):
                    ds, de = 0, min(doc_rows, D)
                    db_np = np.array(d_np[ds:de])
                    next_db = _to_device_chunk(db_np, device, tdtype)

            for ds in range(0, D, doc_rows):
                de = min(ds + doc_rows, D)

                if stream:
                    # Wait for the prefetched chunk to be ready
                    torch.cuda.current_stream().wait_stream(stream)
                    db = next_db

                    # Start prefetching the next chunk in the background
                    next_ds = ds + doc_rows
                    if next_ds < D:
                        with torch.cuda.stream(stream):
                            next_de = min(next_ds + doc_rows, D)
                            db_np = np.array(d_np[next_ds:next_de])
                            next_db = _to_device_chunk(db_np, device, tdtype)
                else:  # CPU path
                    db = _to_device_chunk(np.array(d_np[ds:de]), device, tdtype)

                scores = torch.matmul(qb, db.t().contiguous())

                # Partial top-k within this chunk. Use float32 for stability.
                chunk_k = min(k, de - ds)
                vals, idx = torch.topk(scores.float(), k=chunk_k, dim=1, largest=True, sorted=True)
                idx = idx + ds  # adjust indices to global doc ids

                # Move results to CPU to free up VRAM for the next chunk
                vals_cpu = vals.cpu()
                idx_cpu = idx.cpu()

                if prev_scores is None:
                    prev_scores, prev_indices = vals_cpu, idx_cpu
                else:
                    # Merge previous top-k with current chunk top-k on the CPU
                    combined_scores = torch.cat([prev_scores, vals_cpu], dim=1)
                    combined_indices = torch.cat([prev_indices, idx_cpu], dim=1)
                    if combined_scores.shape[1] > k:
                        prev_scores, pick = torch.topk(combined_scores, k=k, dim=1, largest=True, sorted=True)
                        prev_indices = torch.gather(combined_indices, 1, pick)
                    else:
                        prev_scores, prev_indices = combined_scores, combined_indices

                # Free tensors quickly (let caching allocator reuse memory)
                del db, scores, vals, idx
                if device.type == "cuda" and empty_cache_mode == "chunk":
                    torch.cuda.empty_cache()

            # Save batch results to CPU arrays
            topk_scores[qs:qe] = prev_scores.detach().cpu().numpy()
            topk_indices[qs:qe] = prev_indices.detach().cpu().numpy()

            # Free batch tensors
            del qb, prev_scores, prev_indices
            if device.type == "cuda" and empty_cache_mode in ("batch", "chunk"):
                torch.cuda.empty_cache()

            done = qe
            if Q > 0:
                pct = 100.0 * done / Q
            else:
                pct = 100.0
            print(f"Processed queries {qs}:{qe} ({pct:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")

    # Persist results
    if out_path is None:
        out_path = _derive_output_path(q_path, d_path, k)

    out_obj = {
        "indices": topk_indices,  # shape (Q, K), int64
        "scores": topk_scores,    # shape (Q, K), float32
        "queries_path": q_path,
        "docs_path": d_path,
        "k": k,
        "batch_size": batch_size,
        "dtype": str(q_np.dtype),
        "device": device.type,
    }

    with open(out_path, "wb") as f:
        pickle.dump(out_obj, f)

    print(f"Saved: {out_path}")
    return out_path