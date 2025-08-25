import torch
import os
from tiny_knn.api import exact_search
import contextlib
import pytest
import torch.nn.functional as F

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

# Existing tests

def test_exact_search_ip():
    Q = 10
    D = 100
    dim = 16
    k = 5

    queries = torch.randn(Q, dim, dtype=torch.float32)
    docs = torch.randn(D, dim, dtype=torch.float32)

    indices, scores = exact_search(queries, docs, k, metric="ip")

    expected_scores, expected_indices = torch.topk(torch.matmul(queries, docs.t()), k=k)
    assert torch.allclose(scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_exact_search_cosine():
    Q = 10
    D = 100
    dim = 16
    k = 5

    queries = torch.randn(Q, dim, dtype=torch.float32)
    docs = torch.randn(D, dim, dtype=torch.float32)

    indices, scores = exact_search(queries, docs, k, metric="cosine")

    q_norm = F.normalize(queries, p=2, dim=1)
    d_norm = F.normalize(docs, p=2, dim=1)
    expected_scores, expected_indices = torch.topk(torch.matmul(q_norm, d_norm.t()), k=k)

    assert torch.allclose(scores, expected_scores, atol=1e-6)
    assert torch.equal(indices, expected_indices)


# New tests

def test_k_validation_raises():
    q = torch.randn(4, 8)
    d = torch.randn(3, 8)
    with pytest.raises(ValueError):
        exact_search(q, d, k=5, metric='ip')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_bruteforce_tiny_dtypes(dtype):
    Q, D, dim, k = 5, 7, 4, 3
    q = torch.randn(Q, dim, dtype=dtype)
    d = torch.randn(D, dim, dtype=dtype)
    indices, scores = exact_search(q, d, k=k, metric='ip')
    # Reference on CPU in fp32 for stability
    ref = (q.to(torch.float32) @ d.to(torch.float32).t())
    vals, idx = torch.topk(ref, k=k, dim=1, largest=True, sorted=True)
    assert torch.allclose(scores, vals, atol=1e-3, rtol=1e-3)
    assert torch.equal(indices, idx)


def test_tie_handling_basic():
    # Create exact ties
    q = torch.tensor([[1., 0.]], dtype=torch.float32)
    d = torch.tensor([[1., 0.],[1., 0.],[0., 1.]], dtype=torch.float32)
    indices, scores = exact_search(q, d, k=2, metric='ip')
    assert torch.allclose(scores[0], torch.tensor([1.,1.]))
    # Accept either [0,1] or [1,0] as ties may be unordered, but should be from the tied set {0,1}
    assert set(indices[0].tolist()) == {0, 1}


def test_returns_same_type_for_numpy(tmp_path):
    # When inputs are .npy files, outputs are numpy arrays
    import numpy as np
    q = (np.random.rand(6, 3).astype(np.float32) * 2) - 1
    d = (np.random.rand(7, 3).astype(np.float32) * 2) - 1
    q_path = tmp_path / "q.npy"
    d_path = tmp_path / "d.npy"
    np.save(q_path, q)
    np.save(d_path, d)
    indices, scores = exact_search(str(q_path), str(d_path), k=2, metric='ip')
    assert isinstance(indices, np.ndarray)
    assert isinstance(scores, np.ndarray)

    os.remove(q_path)
    os.remove(d_path)
