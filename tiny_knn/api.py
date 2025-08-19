import os
import time
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def _estimate_batch_size(Q: int, dim: int, emb_bytes: int, safety_frac: float = 0.5, max_batch_size: int = 131072) -> int:
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


def _estimate_doc_chunk_rows(D: int, dim: int, batch_size: int, emb_bytes: int, score_bytes: int, safety_frac: float = 0.5, max_chunk_size: int = 131072) -> int:
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


def exact_search(
    q_path: str,
    d_path: str,
    k: int,
    metric: str | None = None,
    normalize: bool | None = None,
    out_path: str | None = None,
    device: str | torch.device | None = None,
) -> str:
    """
    Exact top-K search with chunking to avoid OOM. Supports:
      - metric='ip' (inner product)
      - metric='cosine' (L2-normalized dot-product)
    Backwards-compat: `normalize=True` forces cosine behavior.
    Inputs and outputs are torch-saved (.pt).
    """
    start_time = time.time()

    # Load tensors (CPU) with optional NumPy memmap for large .npy files
    use_numpy = q_path.endswith('.npy') or d_path.endswith('.npy')
    if use_numpy:
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise ImportError("NumPy is required to read .npy files. Please install numpy.") from e
        q_np = np.load(q_path, mmap_mode='r')
        d_np = np.load(d_path, mmap_mode='r')
        if q_np.ndim != 2 or d_np.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays of shape (N, dim)")
        if q_np.shape[1] != d_np.shape[1]:
            raise ValueError(f"Dim mismatch: queries dim={q_np.shape[1]} vs docs dim={d_np.shape[1]}")
        if q_np.dtype != d_np.dtype:
            raise ValueError(f"Dtype mismatch: queries dtype={q_np.dtype} vs docs dtype={d_np.dtype}")
        # Map NumPy dtype to torch dtype
        if q_np.dtype == np.float32:
            tdtype = torch.float32
        elif q_np.dtype == np.float16:
            tdtype = torch.float16
        else:
            raise ValueError(f"Unsupported NumPy dtype: {q_np.dtype}. Use float32 or float16.")
        Q, dim = q_np.shape
        D, _ = d_np.shape
    else:
        q_cpu: torch.Tensor = torch.load(q_path, map_location="cpu")
        d_cpu: torch.Tensor = torch.load(d_path, map_location="cpu")
        if q_cpu.dim() != 2 or d_cpu.dim() != 2:
            raise ValueError("Embeddings must be 2D tensors of shape (N, dim)")
        if q_cpu.shape[1] != d_cpu.shape[1]:
            raise ValueError(f"Dim mismatch: queries dim={q_cpu.shape[1]} vs docs dim={d_cpu.shape[1]}")
        if q_cpu.dtype != d_cpu.dtype:
            raise ValueError(f"Dtype mismatch: queries dtype={q_cpu.dtype} vs docs dtype={d_cpu.dtype}")
        if q_cpu.dtype is torch.int8 or d_cpu.dtype is torch.int8:
            raise ValueError("int8 dtype is not supported. Cast inputs to float16, bfloat16, or float32.")
        Q, dim = q_cpu.shape
        D, _ = d_cpu.shape
        tdtype = q_cpu.dtype

    # Validate k
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if k > D:
        raise ValueError(f"k={k} exceeds number of docs D={D}")

    # Resolve device
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    # Reproducibility toggle
    det_flag = os.environ.get("TINYKNN_DETERMINISTIC", "0").lower() in ("1", "true", "yes", "on")
    if det_flag:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        if torch_device.type == "cuda":
            # Must be set before first cuBLAS call
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Backend/precision options
    if torch_device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        try:
            cpu_threads = int(os.environ.get("TINYKNN_CPU_THREADS", "0"))
            if cpu_threads > 0:
                torch.set_num_threads(cpu_threads)
        except Exception:
            pass

    # Determine operation type
    metric = (metric or "ip").lower()
    if normalize is None:
        normalize_flag = (metric == "cosine")
    else:
        normalize_flag = bool(normalize)

    emb_bytes = torch.tensor([], dtype=tdtype).element_size()
    score_bytes = 4  # fp32 scores

    # Heuristics
    batch_size = _estimate_batch_size(Q, dim, emb_bytes)
    doc_rows = _estimate_doc_chunk_rows(D, dim, batch_size, emb_bytes, score_bytes)

    empty_cache_mode = os.environ.get("TINYKNN_EMPTY_CACHE", "batch")

    # Outputs on CPU
    topk_scores = torch.empty((Q, k), dtype=torch.float32)
    topk_indices = torch.empty((Q, k), dtype=torch.int64)

    print(
        f"Device={torch_device.type}, metric={'cosine' if normalize_flag else 'ip'}, Q={Q}x{dim}, D={D}x{dim}, batch_size={batch_size}, doc_rows={doc_rows}, k={k}, dtype={tdtype}")

    with torch.inference_mode():
        stream = torch.cuda.Stream() if torch_device.type == "cuda" else None

        for qs in range(0, Q, batch_size):
            qe = min(qs + batch_size, Q)

            # Move query batch
            if use_numpy:
                qb_cpu = torch.from_numpy(np.asarray(q_np[qs:qe]))
            else:
                qb_cpu = q_cpu[qs:qe]
            qb = _to_device_chunk(qb_cpu, torch_device, tdtype)

            # CPU matmul stability for low precision
            cpu_lowp = (torch_device.type == "cpu" and qb.dtype in (torch.float16, torch.bfloat16))
            if normalize_flag:
                if cpu_lowp:
                    qb = F.normalize(qb.to(torch.float32), p=2, dim=1, eps=1e-12)
                else:
                    qb = F.normalize(qb, p=2, dim=1, eps=1e-12)

            prev_scores = None
            prev_indices = None

            # Prefetch first chunk
            next_db = None
            if stream:
                with torch.cuda.stream(stream):
                    ds0, de0 = 0, min(doc_rows, D)
                    if use_numpy:
                        db_cpu0 = torch.from_numpy(np.asarray(d_np[ds0:de0]))
                    else:
                        db_cpu0 = d_cpu[ds0:de0]
                    chunk0 = _to_device_chunk(db_cpu0, torch_device, tdtype)
                    if normalize_flag:
                        if torch_device.type == "cpu" and chunk0.dtype in (torch.float16, torch.bfloat16):
                            chunk0 = F.normalize(chunk0.to(torch.float32), p=2, dim=1, eps=1e-12)
                        else:
                            chunk0 = F.normalize(chunk0, p=2, dim=1, eps=1e-12)
                    next_db = chunk0

            for ds in range(0, D, doc_rows):
                de = min(ds + doc_rows, D)

                if stream:
                    torch.cuda.current_stream().wait_stream(stream)
                    db = next_db
                    # Prefetch next
                    ns = ds + doc_rows
                    if ns < D:
                        with torch.cuda.stream(stream):
                            ne = min(ns + doc_rows, D)
                            if use_numpy:
                                db_cpun = torch.from_numpy(np.asarray(d_np[ns:ne]))
                            else:
                                db_cpun = d_cpu[ns:ne]
                            chunk = _to_device_chunk(db_cpun, torch_device, tdtype)
                            if normalize_flag:
                                if torch_device.type == "cpu" and chunk.dtype in (torch.float16, torch.bfloat16):
                                    chunk = F.normalize(chunk.to(torch.float32), p=2, dim=1, eps=1e-12)
                                else:
                                    chunk = F.normalize(chunk, p=2, dim=1, eps=1e-12)
                            next_db = chunk
                else:
                    if use_numpy:
                        db_cpu = torch.from_numpy(np.asarray(d_np[ds:de]))
                    else:
                        db_cpu = d_cpu[ds:de]
                    db = _to_device_chunk(db_cpu, torch_device, tdtype)
                    if normalize_flag:
                        if torch_device.type == "cpu" and db.dtype in (torch.float16, torch.bfloat16):
                            db = F.normalize(db.to(torch.float32), p=2, dim=1, eps=1e-12)
                        else:
                            db = F.normalize(db, p=2, dim=1, eps=1e-12)

                # Matmul with proper numerics
                if cpu_lowp:
                    scores = torch.matmul(qb.to(torch.float32), db.to(torch.float32).t().contiguous())
                elif torch_device.type == "cuda" and qb.dtype in (torch.float16, torch.bfloat16):
                    # Use modern autocast API; GEMM uses FP16/BF16 inputs with FP32 accumulate
                    with torch.amp.autocast(device_type='cuda', dtype=qb.dtype):
                        scores = torch.matmul(qb, db.t().contiguous())
                else:
                    scores = torch.matmul(qb, db.t().contiguous())

                chunk_k = min(k, de - ds)
                vals, idx = torch.topk(scores.float(), k=chunk_k, dim=1, largest=True, sorted=True)
                idx = idx + ds

                vals_cpu, idx_cpu = vals.cpu(), idx.cpu()

                if prev_scores is None:
                    prev_scores, prev_indices = vals_cpu, idx_cpu
                else:
                    combined_scores = torch.cat([prev_scores, vals_cpu], dim=1)
                    combined_indices = torch.cat([prev_indices, idx_cpu], dim=1)
                    if combined_scores.shape[1] > k:
                        prev_scores, pick = torch.topk(combined_scores, k=k, dim=1, largest=True, sorted=True)
                        prev_indices = torch.gather(combined_indices, 1, pick)
                    else:
                        prev_scores, prev_indices = combined_scores, combined_indices

                del db, scores, vals, idx
                if torch_device.type == "cuda" and empty_cache_mode == "chunk":
                    torch.cuda.empty_cache()

            topk_scores[qs:qe] = prev_scores
            topk_indices[qs:qe] = prev_indices

            del qb, prev_scores, prev_indices
            if torch_device.type == "cuda" and empty_cache_mode in ("batch", "chunk"):
                torch.cuda.empty_cache()

            pct = 100.0 * min(qe, Q) / max(Q, 1)
            print(f"Processed queries {qs}:{qe} ({pct:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")

    if out_path is None:
        out_path = _derive_output_path(q_path, d_path, k)

    out_obj = {
        "indices": topk_indices,
        "scores": topk_scores,
        "queries_path": q_path,
        "docs_path": d_path,
        "k": k,
        "batch_size": batch_size,
        "dtype": str(tdtype),
        "device": torch_device.type,
        "metric": "cosine" if normalize_flag else "ip",
    }
    torch.save(out_obj, out_path)
    print(f"Saved: {out_path}")
    return out_path
