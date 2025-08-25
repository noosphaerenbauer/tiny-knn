import argparse
import os
import numpy as np
import torch
from .api import exact_search


def _save_results(indices, scores, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".pt", ".pth"):
        if isinstance(indices, np.ndarray):
            indices_t = torch.from_numpy(indices)
            scores_t = torch.from_numpy(scores)
        else:
            indices_t = indices
            scores_t = scores
        torch.save({"indices": indices_t, "scores": scores_t}, out_path)
    elif ext == ".npz":
        if isinstance(indices, torch.Tensor):
            indices_np = indices.cpu().numpy()
            scores_np = scores.cpu().numpy()
        else:
            indices_np = indices
            scores_np = scores
        np.savez(out_path, indices=indices_np, scores=scores_np)
    else:
        # Default to torch save
        if isinstance(indices, np.ndarray):
            indices_t = torch.from_numpy(indices)
            scores_t = torch.from_numpy(scores)
        else:
            indices_t = indices
            scores_t = scores
        torch.save({"indices": indices_t, "scores": scores_t}, out_path)


def main():
    parser = argparse.ArgumentParser(description="Compute top-K similarities between query and document embeddings.")
    parser.add_argument("queries_path", help="Path to queries (.pt or .npy)")
    parser.add_argument("docs_path", help="Path to documents (.pt or .npy)")
    parser.add_argument("-k", "--k", type=int, default=100, help="Top-K per query to keep")
    parser.add_argument("--metric", choices=["ip", "cosine"], default="ip", help="Similarity metric")
    parser.add_argument("--output-path", help="Optional path to save results (.pt or .npz)")

    args = parser.parse_args()

    indices, scores = exact_search(args.queries_path, args.docs_path, args.k, metric=args.metric)

    if args.output_path:
        _save_results(indices, scores, args.output_path)
    else:
        print(f"Computed top-{args.k}: indices shape={getattr(indices, 'shape', None)}, scores shape={getattr(scores, 'shape', None)}")

if __name__ == "__main__":
    main()
