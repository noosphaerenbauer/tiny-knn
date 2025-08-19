import argparse
from .api import exact_search

def main():
    parser = argparse.ArgumentParser(description="Compute top-K similarities between query and document embeddings.")
    parser.add_argument("queries_path", help="Path to queries .pt file")
    parser.add_argument("docs_path", help="Path to documents .pt file")
    parser.add_argument("--output-path", help="Path to output .pt file")
    parser.add_argument("-k", "--k", type=int, default=100, help="Top-K per query to keep")
    
    # Performance & hardware
    parser.add_argument("--batch-size", type=int, default=1024, help="Query batch size")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Document chunk size")
    parser.add_argument("--autotune", action="store_true", help="Automatically estimate batch and chunk sizes")
    parser.add_argument("--device", default="cuda", help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu', 'mps')")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Computation data type")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loading")

    # Algorithm
    parser.add_argument("--normalize", action="store_true", help="Normalize vectors to unit length (cosine similarity)")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except for errors")

    # Algorithm
    parser.add_argument("--normalize", action="store_true", help="Normalize vectors to unit length (cosine similarity)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except for errors")

    args = parser.parse_args()

    exact_search(
        q_path=args.queries_path,
        d_path=args.docs_path,
        k=args.k,
        normalize=args.normalize,
        deterministic=args.deterministic,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        autotune=args.autotune,
        num_workers=args.num_workers,
        out_path=args.output_path,
        verbose=args.verbose,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
