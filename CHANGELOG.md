# Changelog

All notable changes to this project are documented here.

## [0.2.0] - 2025-08-25

- API: `exact_search` signature changed to `exact_search(arr1, arr2, k, metric)` and now returns a tuple `(indices, scores)` instead of writing to disk.
- I/O: Accepts Torch tensors, NumPy arrays, or file paths to `.pt`/`.npy`; returns results in the same type family (Torch or NumPy).
- Metrics: `metric` is required as `'ip'` or `'cosine'`; cosine does internal L2-normalization.
- CLI: Simplified interface. New flags `--metric {ip,cosine}` and `--output-path`. Removed low-level controls (device/dtype/batch/chunk/autotune/etc.).
- CLI output: When `--output-path` is provided, saves results as `.pt` or `.npz` with keys `indices` and `scores`.
- Benchmarks: Updated to the new function signature and direct-return behavior.
- Tests: Rewritten to validate tuple returns and type-family behavior.
- Docs: README overhauled to reflect the new API and CLI usage.

Migration notes:
- Previous code using `exact_search(q_path=..., d_path=..., out_path=...)` must switch to direct arguments and handle the return values: `indices, scores = exact_search(..., metric='ip')`.
- For CLI, replace `--normalize` with `--metric cosine`. Use `--output-path` to save results.

## [0.1.0]

- Initial release.
