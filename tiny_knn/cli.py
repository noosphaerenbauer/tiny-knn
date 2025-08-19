import argparse
from .api import compute_topk

def main():
    parser = argparse.ArgumentParser(description="Compute top-K similarities (dot-product) between query and document embeddings using GPU batching.")
    parser.add_argument("queries_path", help="Path to queries .npy file")
    parser.add_argument("docs_path", help="Path to documents .npy file")
    parser.add_argument("--k", type=int, default=2000, help="Top-K per query to keep")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (fallback if no CUDA)")
    parser.add_argument("--output-path", help="Path to output .pkl file")
    args = parser.parse_args()

    compute_topk(
        q_path=args.queries_path,
        d_path=args.docs_path,
        k=args.k,
        force_cpu=args.cpu,
        out_path=args.output_path,
    )

if __name__ == "__main__":
    main()
