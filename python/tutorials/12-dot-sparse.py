"""NVIDIA 2:4 sparsity tutorial with dedicated SparseMetadataEncodingAttr.

Supports multi-warp and large tile sizes. Metadata uses its own distributed
encoding which allows natural N-warp duplication (unlike LinearEncoding).
"""
import random
import numpy as np
import torch
import triton
import triton.language as tl

torch.manual_seed(42)
random.seed(42)


@triton.jit
def sparse_matmul_kernel(
    a_ptr, b_ptr, c_ptr, meta_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_mm, stride_mk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_a = tl.arange(0, BLOCK_K // 2)
    offs_k_b = tl.arange(0, BLOCK_K)
    offs_k_m = tl.arange(0, BLOCK_K // 16)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_a[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k_b[:, None] * stride_bk + offs_n[None, :] * stride_bn
    m_ptrs = meta_ptr + offs_m[:, None] * stride_mm + offs_k_m[None, :] * stride_mk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        meta = tl.load(m_ptrs)
        acc = tl.dot_sparse(a, b, meta, acc)
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        m_ptrs += (BLOCK_K // 16) * stride_mk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def make_sparse(A):
    # fp16 randn can produce exact zeros; ensure all elements are non-zero
    # before pruning so that compress() can identify kept positions by != 0
    A[A == 0] = torch.finfo(A.dtype).tiny
    for i in range(A.shape[0]):
        indices = []
        for j in range(0, A.shape[1], 4):
            indices.extend(random.sample([j, j + 1, j + 2, j + 3], 2))
        A[i, indices] = 0
    return A


def compress(A):
    flat = A.flatten().cpu().detach().numpy()
    nonzero_indices = []
    meta_nibbles = []
    for outerIdx in range(0, len(flat), 4):
        nibble = 0
        nzCount = 0
        for innerIdx in range(4):
            if flat[outerIdx + innerIdx] != 0:
                nonzero_indices.append(outerIdx + innerIdx)
                nibble |= innerIdx << (2 * nzCount)
                nzCount += 1
        assert nzCount == 2
        meta_nibbles.append(nibble)
    metas = []
    for outerIdx in range(0, len(meta_nibbles), 4):
        meta = 0
        for i in range(4):
            meta |= meta_nibbles[outerIdx + i] << (4 * i)
        metas.append(meta)
    aSparse = flat[nonzero_indices].reshape(A.shape[0], A.shape[1] // 2)
    aMeta = np.array(metas, dtype=np.uint16).astype(np.int16).reshape(
        A.shape[0], A.shape[1] // 16)
    return (torch.tensor(aSparse, device=A.device, dtype=A.dtype),
            torch.tensor(aMeta, device=A.device))


def matmul_sparse(a_sparse, b, a_meta, M, N, K,
                  BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4):
    c = torch.zeros(M, N, device=b.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sparse_matmul_kernel[grid](
        a_sparse, b, c, a_meta,
        M, N, K,
        a_sparse.stride(0), a_sparse.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_meta.stride(0), a_meta.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return c


def test_correctness(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    a_dense = make_sparse(torch.randn(M, K, device='cuda', dtype=torch.float16))
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    a_sparse, a_meta = compress(a_dense)
    c = matmul_sparse(a_sparse, b, a_meta, M, N, K,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                      num_warps=num_warps)
    ref = torch.matmul(a_dense, b)
    diff = (c.float() - ref.float()).abs().max().item()
    close = torch.allclose(c.float(), ref.float(), atol=0.5, rtol=0.1)
    mark = "PASS" if close else "FAIL"
    print(f"[{mark}] M={M:4d} N={N:4d} K={K:4d}  BLK=({BLOCK_M:3d},{BLOCK_N:3d},{BLOCK_K:3d}) "
          f"nwarps={num_warps}: max_diff={diff:.6f}")


def bench_one(label, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
              warmup=50, repeat=200):
    a_dense = make_sparse(torch.randn(M, K, device='cuda', dtype=torch.float16))
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    a_sparse, a_meta = compress(a_dense)

    for _ in range(warmup):
        torch.matmul(a_dense, b)
        matmul_sparse(a_sparse, b, a_meta, M, N, K,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                      num_warps=num_warps)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        torch.matmul(a_dense, b)
    end.record()
    torch.cuda.synchronize()
    dense_us = start.elapsed_time(end) * 1000 / repeat

    start.record()
    for _ in range(repeat):
        matmul_sparse(a_sparse, b, a_meta, M, N, K,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                      num_warps=num_warps)
    end.record()
    torch.cuda.synchronize()
    sparse_us = start.elapsed_time(end) * 1000 / repeat

    flops = 2 * M * K * N
    dense_tflops = flops / (dense_us * 1e-6) / 1e12
    sparse_tflops = flops / (sparse_us * 1e-6) / 1e12
    speedup = dense_us / sparse_us
    return label, M, K, N, dense_us, sparse_us, speedup, dense_tflops, sparse_tflops


if __name__ == "__main__":
    torch.manual_seed(42)
    gpu_name = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability(0)
    print(f"Triton 2:4 Sparse vs Dense cuBLAS Benchmark ({gpu_name}, SM{sm[0]}{sm[1]})\n")

    print("  --- Correctness (various block sizes & warp counts) ---")
    correctness_configs = [
        # (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        (16, 16, 32, 16, 16, 32, 1),    # single MMA tile
        (32, 32, 64, 32, 32, 64, 1),    # 2x2 MMA tiles
        (64, 64, 64, 32, 32, 64, 1),
        (64, 64, 64, 64, 64, 64, 2),    # multi-warp
        (128, 128, 128, 64, 64, 64, 4),  # multi-warp + multi-tile
        (256, 256, 256, 64, 64, 64, 4),
        (512, 512, 512, 128, 128, 64, 4),
    ]
    for cfg in correctness_configs:
        test_correctness(*cfg)

    # -- Benchmark --
    # (label, M, K, N, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    large = [
        # large square
        ("square 2048",        2048, 2048, 2048,   128, 128, 64, 4),
        ("square 4096",        4096, 4096, 4096,   128, 128, 64, 4),
        # LLaMA-7B
        ("LLaMA-7B QKV",       4096, 4096, 4096,   128, 128, 64, 4),
        ("LLaMA-7B FFN up",    4096, 4096, 11008,  128, 128, 64, 4),
        ("LLaMA-7B FFN down",  4096, 11008, 4096,  128, 128, 64, 4),
        # LLaMA-13B
        ("LLaMA-13B FFN up",   5120, 5120, 13824,  128, 128, 64, 4),
        ("LLaMA-13B FFN down", 5120, 13824, 5120,  128, 128, 64, 4),
        # large batch
        ("batch 512",          512, 4096, 4096,    64, 64, 64, 4),
        ("batch 1024",         1024, 4096, 4096,   128, 128, 64, 4),
        ("batch 2048",         2048, 4096, 4096,   128, 128, 64, 4),
    ]
    small = [
        # small square
        ("square 1024",        1024, 1024, 1024,   128, 128, 64, 4),
        # small batch: less work per tile, so the sparse win shrinks
        ("batch 32",           32, 4096, 4096,     32, 64, 64, 2),
        ("batch 64",           64, 4096, 4096,     64, 64, 64, 2),
        ("batch 128",          128, 4096, 4096,    64, 64, 64, 4),
        ("batch 256",          256, 4096, 4096,    64, 64, 64, 4),
    ]

    header = f"  {'label':<20} {'M':>6} {'K':>6} {'N':>6} | {'Dense (us)':>11} {'Sparse (us)':>12} {'Speedup':>8} | {'Dense TFLOPS':>13} {'Sparse TFLOPS':>14}"
    sep = "-" * len(header)
    row_fmt = "  {:<20} {:>6} {:>6} {:>6} | {:>10.1f} {:>11.1f} {:>7.2f}x | {:>12.1f} {:>13.1f}"

    for title, cases in [("LARGE (M, K, N)", large),
                          ("SMALL (small batch / small matrix)", small)]:
        print(f"\n  {title}")
        print(header)
        print(sep)
        for label, M, K, N, BM, BN, BK, nw in cases:
            _, _, _, _, du, su, sp, dt, st = bench_one(label, M, N, K, BM, BN, BK, nw)
            print(row_fmt.format(label, M, K, N, du, su, sp, dt, st), flush=True)
