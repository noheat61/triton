#!/usr/bin/env python3
"""Does tl.dot_sparse reach Triton's Blackwell fast path, and is that path faster?

Two questions, and they have different answers.

1. **Does it work?** `tl.dot_sparse` inside a persistent loop over TMA tensor
   descriptors with `warp_specialize=True` lowers to `tcgen05.mma.sp` and gives
   exact results. The metadata rides a descriptor of its own; nothing in the
   warp-specialization or TMA machinery has to know it is there. Worth pinning
   down, because that path is what cuBLAS-competitive Blackwell kernels use and
   a sparse dot that could not enter it would be a dead end.

2. **Is it faster?** Not on every part. This script measures the TMA path
   against the plain `tl.load` + L2-swizzle kernel and against cuBLAS, all in
   one process with the three winners re-timed interleaved, because clock drift
   between separate runs on this class of device is larger than the effect.

The TMA kernels here are hand-written and lightly tuned: no epilogue subtiling,
no 2-CTA mode, no CLC scheduling. Read a slow TMA row as "this shape and this
part did not reward the persistent design as written", not as a verdict on TMA.
"""
import argparse
import statistics
import sys

import torch
import triton
import triton.language as tl
import triton.testing as tt


@triton.jit
def _pid(tile, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    group_id = tile // num_pid_in_group
    first = group_id * GROUP_M
    group_size = min(num_pid_m - first, GROUP_M)
    return first + ((tile % num_pid_in_group) % group_size), (tile % num_pid_in_group) // group_size


@triton.jit
def sparse_tma_kernel(a_ptr, b_ptr, c_ptr, m_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
                      WARP_SPECIALIZE: tl.constexpr):
    start = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # B is described as [N, K] and transposed at the dot, which is what the
    # descriptor path wants; the sparse lhs and its metadata are [M, K/2] and
    # [M, K/16]. BLOCK_K >= 128 keeps the metadata's innermost block at or above
    # the 16 bytes a descriptor needs.
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // 2], [K // 2, 1], [BLOCK_M, BLOCK_K // 2])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    m_desc = tl.make_tensor_descriptor(m_ptr, [M, K // 16], [K // 16, 1], [BLOCK_M, BLOCK_K // 16])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    num_pid_in_group = GROUP_M * num_pid_n
    for tile in tl.range(start, num_pid_m * num_pid_n, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _pid(tile, num_pid_in_group, num_pid_m, GROUP_M)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            off_k = ki * BLOCK_K
            acc = tl.dot_sparse(a_desc.load([off_m, off_k // 2]), b_desc.load([off_n, off_k]).T,
                                m_desc.load([off_m, off_k // 16]), acc)
        c_desc.store([off_m, off_n], acc.to(c_ptr.dtype.element_ty))


@triton.jit
def dense_tma_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
                     WARP_SPECIALIZE: tl.constexpr):
    start = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    num_pid_in_group = GROUP_M * num_pid_n
    for tile in tl.range(start, num_pid_m * num_pid_n, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _pid(tile, num_pid_in_group, num_pid_m, GROUP_M)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            off_k = ki * BLOCK_K
            acc = tl.dot(a_desc.load([off_m, off_k]), b_desc.load([off_n, off_k]).T, acc)
        c_desc.store([off_m, off_n], acc.to(c_ptr.dtype.element_ty))


CONFIGS = [(128, 128, 128, 8), (128, 256, 128, 8), (256, 128, 128, 8), (128, 128, 256, 8), (128, 128, 128, 4),
           (256, 256, 128, 8)]
GROUP_MS = [1, 8]


def tf(ms, M, N, K):
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def warm_up():
    import time
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        for _ in range(10):
            x = torch.matmul(x, x) * 0 + x
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4096,8192")
    ap.add_argument("--check", action="store_true", help="verify the sparse TMA kernel against a dense reference")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        print("tensor descriptors + warp specialization need Blackwell", file=sys.stderr)
        return 2

    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    print(f"device : {torch.cuda.get_device_name()} (sm_{major * 10 + minor}, {num_sms} SMs)")

    if args.check:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from verify_dot_sparse import compress_24, make_24_operands
        M = N = K = 512
        a_dense, b_dense = make_24_operands(M, N, K, [-2.0, -1.0, 0.5, 1.0, 1.5, 2.0], "cuda")
        a_sparse, meta = compress_24(a_dense)
        c = torch.zeros((M, N), device="cuda", dtype=torch.float16)
        for ws in (False, True):
            sparse_tma_kernel[(min(num_sms, 16), )](a_sparse.to(torch.float16),
                                                    b_dense.T.contiguous().to(torch.float16), c, meta, M, N, K,
                                                    BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                                                    NUM_SMS=min(num_sms, 16), WARP_SPECIALIZE=ws, num_warps=8)
            err = (c.to(torch.float64) - torch.matmul(a_dense, b_dense)).abs().max().item()
            print(f"  warp_specialize={ws}: max abs err {err:.3e}  {'OK' if err == 0 else 'MISMATCH'}")

    warm_up()
    for size in (int(s) for s in args.sizes.split(",")):
        run(size, num_sms)
    return 0


def run(S, num_sms):
    M = N = K = S
    a = torch.randn(M, K // 2, device="cuda", dtype=torch.float16)
    a_full = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b_nk = torch.randn(N, K, device="cuda", dtype=torch.float16)
    b_kn = b_nk.T.contiguous()
    meta = torch.full((M, K // 16), 0x4444, device="cuda", dtype=torch.int16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    def launch(cfg, sparse):
        bm, bn, bk, w, gm, ws = cfg
        grid = (min(num_sms, triton.cdiv(M, bm) * triton.cdiv(N, bn)), )
        if sparse:
            return lambda: sparse_tma_kernel[grid](a, b_nk, c, meta, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                                   GROUP_M=gm, NUM_SMS=num_sms, WARP_SPECIALIZE=ws, num_warps=w)
        return lambda: dense_tma_kernel[grid](a_full, b_nk, c, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              GROUP_M=gm, NUM_SMS=num_sms, WARP_SPECIALIZE=ws, num_warps=w)

    winners = {}
    for label, sparse in (("sparse", True), ("dense ", False)):
        best = (0.0, None)
        for base in CONFIGS:
            for gm in GROUP_MS:
                for ws in (False, True):
                    cfg = (*base, gm, ws)
                    try:
                        v = tf(tt.do_bench(launch(cfg, sparse), warmup=25, rep=100), M, N, K)
                    except Exception:  # noqa: BLE001  (a config this device cannot fit)
                        continue
                    if v > best[0]:
                        best = (v, cfg)
        winners[label] = best[1]

    # Re-time the winners interleaved with the library kernel. Measuring them in
    # sequence instead lets the clock drift show up as a speedup.
    rows = []
    for _ in range(3):
        rows.append((
            tf(tt.do_bench(launch(winners["sparse"], True), warmup=25, rep=100), M, N, K),
            tf(tt.do_bench(launch(winners["dense "], False), warmup=25, rep=100), M, N, K),
            max(tf(tt.do_bench(lambda: torch.matmul(a_full, b_kn), warmup=25, rep=100), M, N, K),
                tf(tt.do_bench(lambda: torch.matmul(a_full, b_nk.T), warmup=25, rep=100), M, N, K)),
        ))
    sp, dn, lib = (statistics.median([r[i] for r in rows]) for i in range(3))
    print(f"\n{S}^3  sparse {sp:6.1f}  dense {dn:6.1f}  torch {lib:6.1f} TF/s"
          f"   sparse/dense {sp / dn:.2f}x  sparse/torch {sp / lib:.2f}x")
    for label, cfg in winners.items():
        print(f"    {label}: block {cfg[0]}x{cfg[1]}x{cfg[2]} w{cfg[3]} group_m {cfg[4]} warp_specialize {cfg[5]}")


if __name__ == "__main__":
    sys.exit(main())
