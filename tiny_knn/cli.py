import argparse
from .api import exact_search

def main():
    parser = argparse.ArgumentParser(description="Compute top-K similarities between query and document embeddings.")
    parser.add_argument("queries_path", help="Path to queries .pt file")
    parser.add_argument("docs_path", help="Path to documents .pt file")
    parser.add_argument("--k", type=int, default=100, help="Top-K per query to keep")
    parser.add_argument("--metric", default="ip", choices=["ip", "cosine"], help="Similarity metric")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (fallback if no CUDA)")
    parser.add_argument("--output-path", help="Path to output .pt file")
    args = parser.parse_args()

    exact_search(
        q_path=args.queries_path,
        d_path=args.docs_path,
        k=args.k,
        metric=args.metric,
        force_cpu=args.cpu,
        out_path=args.output_path,
    )

if __name__ == "__main__":
    main()
