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


# New tests

def test_k_validation_raises():
    with temp_files_context() as temp_files:
        q = torch.randn(4, 8)
        d = torch.randn(3, 8)
        q_path = temp_files(q, 'q')
        d_path = temp_files(d, 'd')
        with pytest.raises(ValueError):
            exact_search(q_path, d_path, k=5, device='cpu')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('use_cuda', [False, True])
def test_bruteforce_tiny_dtype_device(dtype, use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    device = 'cuda' if use_cuda else 'cpu'
    Q, D, dim, k = 5, 7, 4, 3
    q = torch.randn(Q, dim, dtype=dtype)
    d = torch.randn(D, dim, dtype=dtype)
    with temp_files_context() as temp_files:
        q_path = temp_files(q, 'qbf')
        d_path = temp_files(d, 'dbf')
        out_path = exact_search(q_path, d_path, k=k, metric='ip', device=device)
        out = torch.load(out_path)
        # Reference on CPU in fp32 for stability
        ref = (q.to(torch.float32) @ d.to(torch.float32).t())
        vals, idx = torch.topk(ref, k=k, dim=1, largest=True, sorted=True)
        assert torch.allclose(out['scores'], vals, atol=1e-3, rtol=1e-3)
        assert torch.equal(out['indices'], idx)
        if os.path.exists(out_path):
            os.remove(out_path)


def test_tie_handling_determinism():
    # Create exact ties
    q = torch.tensor([[1., 0.]], dtype=torch.float32)
    d = torch.tensor([[1., 0.],[1., 0.],[0., 1.]], dtype=torch.float32)
    with temp_files_context() as temp_files:
        q_path = temp_files(q, 'qt')
        d_path = temp_files(d, 'dt')
        out_path = exact_search(q_path, d_path, k=2, metric='ip', device='cpu')
        out = torch.load(out_path)
        # Expect indices [0,1] with scores [1,1]
        assert torch.allclose(out['scores'][0], torch.tensor([1.,1.]))
        assert torch.equal(out['indices'][0], torch.tensor([0,1]))
        if os.path.exists(out_path):
            os.remove(out_path)


def test_reproducibility_toggle_cpu(monkeypatch):
    monkeypatch.setenv('TINYKNN_DETERMINISTIC', '1')
    q = torch.randn(8, 16)
    d = torch.randn(10, 16)
    with temp_files_context() as temp_files:
        q_path = temp_files(q, 'q2')
        d_path = temp_files(d, 'd2')
        out1_path = exact_search(q_path, d_path, k=4, metric='ip', device='cpu')
        out2_path = exact_search(q_path, d_path, k=4, metric='ip', device='cpu')
        out1 = torch.load(out1_path)
        out2 = torch.load(out2_path)
        assert torch.equal(out1['indices'], out2['indices'])
        assert torch.allclose(out1['scores'], out2['scores'])
        for p in (out1_path, out2_path):
            if os.path.exists(p):
                os.remove(p)
