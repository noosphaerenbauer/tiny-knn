import torch
import os
from tiny_knn.api import exact_search
import contextlib

@contextlib.contextmanager
def temp_files_context():
    files = []
    def create_temp_file(tensor, name):
        path = f"{name}.pt"
        torch.save(tensor, path)
        files.append(path)
        return path
    try:
        yield create_temp_file
    finally:
        for file in files:
            if os.path.exists(file):
                os.remove(file)

def test_exact_search_ip():
    with temp_files_context() as temp_files:
        Q = 10
        D = 100
        dim = 16
        k = 5

        queries = torch.randn(Q, dim, dtype=torch.float32)
        docs = torch.randn(D, dim, dtype=torch.float32)

        q_path = temp_files(queries, "queries")
        d_path = temp_files(docs, "docs")
        out_path = "results.pt"
        files_to_clean = [out_path]

        try:
            result_path = exact_search(q_path, d_path, k, out_path=out_path, device="cpu")
            assert result_path == out_path
            results = torch.load(result_path)

            # Manual calculation
            expected_scores, expected_indices = torch.topk(torch.matmul(queries, docs.t()), k=k)

            assert torch.allclose(results["scores"], expected_scores)
            assert torch.equal(results["indices"], expected_indices)
        finally:
            for f in files_to_clean:
                if os.path.exists(f):
                    os.remove(f)


def test_exact_search_cosine():
    with temp_files_context() as temp_files:
        Q = 10
        D = 100
        dim = 16
        k = 5

        queries = torch.randn(Q, dim, dtype=torch.float32)
        docs = torch.randn(D, dim, dtype=torch.float32)

        q_path = temp_files(queries, "queries")
        d_path = temp_files(docs, "docs")
        out_path = "results.pt"
        files_to_clean = [out_path]

        try:
            result_path = exact_search(q_path, d_path, k, normalize=True, out_path=out_path, device="cpu")
            assert result_path == out_path
            results = torch.load(result_path)

            # Manual calculation
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
            d_norm = torch.nn.functional.normalize(docs, p=2, dim=1)
            expected_scores, expected_indices = torch.topk(torch.matmul(q_norm, d_norm.t()), k=k)

            assert torch.allclose(results["scores"], expected_scores, atol=1e-6)
            assert torch.equal(results["indices"], expected_indices)
        finally:
            for f in files_to_clean:
                if os.path.exists(f):
                    os.remove(f)
