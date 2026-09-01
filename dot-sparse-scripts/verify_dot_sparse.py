#!/usr/bin/env python3
"""Standalone numerical verification for tl.dot_sparse across MMA generations.

Needs only `triton` and `torch` -- no repo checkout, no pytest. Meant for running
on a machine whose GPU this branch has not been validated on yet:

    pip install triton-*.whl torch
    python verify_dot_sparse.py

For every supported dtype and shape it builds a 2:4 sparse lhs whose products are
*exactly* representable in the accumulator, so a mismatch means a wrong metadata
layout rather than rounding. It also reports which sparse instruction the compiler
actually selected, so a silent fallback to an older MMA version is visible.

Exit code is 0 only if every case passed.

On Hopper the sparse path is gated off (its numerics have never run on a Hopper
device); set TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1 to exercise it.
"""
import argparse
import re
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- reference data


def compress_24(a_dense):
    """Split a 2:4 dense matrix into (kept values, int16 metadata).

    This is the executable definition of the metadata format: one i16 packs four
    4-bit groups, each holding the two 2-bit indices of the kept elements of one
    group of four dense elements.
    """
    flat = a_dense.flatten().to(torch.float64).cpu().numpy()
    kept, nibbles = [], []
    for base in range(0, len(flat), 4):
        nibble = count = 0
        for i in range(4):
            if flat[base + i] != 0:
                kept.append(base + i)
                nibble |= i << (2 * count)
                count += 1
        assert count == 2, "every group of four must keep exactly two elements"
        nibbles.append(nibble)
    metas = [sum(nibbles[b + i] << (4 * i) for i in range(4)) for b in range(0, len(nibbles), 4)]
    M, K = a_dense.shape
    a_sparse = torch.tensor(flat[kept], device=a_dense.device).reshape(M, K // 2)
    meta = torch.tensor(np.array(metas, dtype=np.uint16).astype(np.int16), device=a_dense.device)
    return a_sparse, meta.reshape(M, K // 16)


def make_24_operands(M, N, K, values, device, seed=17):
    """Dense lhs with a 2:4 pattern along K, plus a dense rhs, both exact."""
    rs = np.random.RandomState(seed)
    pick = lambda shape: torch.tensor(  # noqa: E731
        np.asarray(values, dtype=np.float64)[rs.randint(len(values), size=shape)], device=device, dtype=torch.float64)
    a_dense = pick((M, K))
    for i in range(M):
        for j in range(0, K, 4):
            drop = rs.choice(4, size=2, replace=False)
            a_dense[i, j + drop[0]] = 0
            a_dense[i, j + drop[1]] = 0
    return a_dense, pick((K, N))


# ---------------------------------------------------------------------- kernel


@triton.jit
def sparse_matmul_kernel(a_ptr, b_ptr, c_ptr, meta_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                         stride_cn, stride_mm, stride_mk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, OUT_DTYPE: tl.constexpr, GROUP_M: tl.constexpr = 1):
    # L2 CTA swizzle. Without it these kernels lose a factor of three at 8192**3
    # (sparse 50 vs 146 TF/s measured on Thor), which swamps any sparse-vs-dense
    # difference and makes this table meaningless. GROUP_M = 1 is the plain
    # row-major order, so correctness callers keep the behaviour they had.
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0) * num_pid_n + tl.program_id(1)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K // 2)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    m_ptrs = meta_ptr + offs_m[:, None] * stride_mm + tl.arange(0, BLOCK_K // 16)[None, :] * stride_mk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=OUT_DTYPE)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot_sparse(tl.load(a_ptrs), tl.load(b_ptrs), tl.load(m_ptrs), acc, out_dtype=OUT_DTYPE)
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        m_ptrs += (BLOCK_K // 16) * stride_mk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def dense_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        OUT_DTYPE: tl.constexpr, GROUP_M: tl.constexpr = 1):
    # L2 CTA swizzle. Without it these kernels lose a factor of three at 8192**3
    # (sparse 50 vs 146 TF/s measured on Thor), which swamps any sparse-vs-dense
    # difference and makes this table meaningless. GROUP_M = 1 is the plain
    # row-major order, so correctness callers keep the behaviour they had.
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0) * num_pid_n + tl.program_id(1)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=OUT_DTYPE)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=OUT_DTYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# --------------------------------------------------------------------- harness

TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "float8e4nv": torch.float8_e4m3fn,
    "float8e5": torch.float8_e5m2,
}

# (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps); K is the dense K.
#
# The first group is small enough to stay on mma.sp everywhere. The rest have
# M % 64 == 0 with num_warps % 4 == 0 so Hopper picks wgmma.mma_async.sp, and
# M >= 128 so datacenter Blackwell picks tcgen05.mma.sp -- without those the run
# would silently only exercise MMAv2. BLOCK_K stays at 64 because a 16-bit
# BLOCK_K=128 tile needs 100 KB of shared memory, over the limit on consumer
# parts; EIGHT_BIT_SHAPES adds wider K where the operands are half as big.
SHAPES = [
    (16, 8, 64, 16, 8, 64, 1),
    (32, 32, 128, 32, 32, 64, 1),
    (32, 32, 128, 32, 32, 128, 1),
    (64, 64, 128, 64, 64, 64, 2),
    (128, 128, 128, 128, 128, 64, 4),
    (128, 256, 256, 128, 128, 64, 4),
    (256, 128, 256, 128, 128, 64, 8),
    # A 256x64 tile on 4 warps lands on warpsPerCTA = [4, 1] for both MMAv2 and
    # MMAv3, and with no N-warps to broadcast over the two metadata layouts are
    # byte-identical. Running it here therefore exercises, bit-exact on this
    # silicon, the very layout Hopper will use -- see the coverage section.
    (256, 64, 128, 256, 64, 64, 4),
]

EIGHT_BIT_SHAPES = [
    (128, 128, 256, 128, 128, 128, 4),
    (256, 256, 512, 128, 256, 128, 8),
]

# `mma.sp` and `mma.sp::ordered_metadata` are both possible (Blackwell takes the
# ordered form), so match the optional `::ordered_metadata` explicitly instead of
# assuming `.sync` follows `.sp`.
SPARSE_INSTR = re.compile(r"(?:tcgen05\.mma\.sp|wgmma\.mma_async\.sp|mma\.sp)"
                          r"(?:::ordered_metadata)?[\w:.]*")


def instr_of(compiled):
    found = sorted(set(SPARSE_INSTR.findall(compiled.asm.get("ptx", ""))))
    return ", ".join(found) if found else "(no sparse instruction!)"


def run_case(dtype, shape, out_dtype, device):
    M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps = shape
    is_int = dtype == "int8"
    values = list(range(-4, 0)) + list(range(1, 5)) if is_int else [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
    a_dense, b_dense = make_24_operands(M, N, K, values, device)
    a_sparse, meta = compress_24(a_dense)
    ref = torch.matmul(a_dense, b_dense)

    a = a_sparse.to(TORCH_DTYPE[dtype])
    b = b_dense.to(TORCH_DTYPE[dtype])
    acc_torch = {tl.int32: torch.int32, tl.float32: torch.float32, tl.float16: torch.float16}[out_dtype]
    c = torch.zeros((M, N), device=device, dtype=acc_torch)

    compiled = sparse_matmul_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        a, b, c, meta, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        meta.stride(0), meta.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, OUT_DTYPE=out_dtype,
        num_warps=num_warps)

    got = c.to(torch.float64)
    if out_dtype is tl.float16:
        # The fp16 accumulator trades precision for speed; the error grows like
        # sqrt(K) and lands around 1e-3 relative for these sizes.
        err = ((got - ref).abs().max() / ref.abs().max().clamp(min=1e-30)).item()
        return err <= 5e-3, f"max rel err {err:.2e}", instr_of(compiled)
    exact = torch.equal(got, ref)
    if exact:
        return True, "bit-exact", instr_of(compiled)
    bad = (got != ref).sum().item()
    return False, f"{bad}/{M * N} elements differ", instr_of(compiled)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). Small on purpose: enough
# for a fair sparse-vs-dense ratio without turning into a tuning run. The
# optimum differs per dtype, which is why both sides get the same sweep --
# comparing a tuned sparse kernel against an untuned dense one inflates the
# ratio, and the printed torch column is there to catch exactly that.
SPEED_CONFIGS = [
    (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 4, 2),
    (128, 128, 128, 4, 2),
    (128, 128, 128, 4, 3),
    # num_warps alone is worth 8x on the dense 8-bit kernel (7 -> 58 TF/s at
    # 128x128x128 on an RTX 3060), so never sweep a 128x128 tile at w4 only.
    (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (128, 256, 128, 8, 3),
    (256, 128, 64, 8, 3),
    (256, 128, 128, 8, 3),
    (256, 256, 64, 8, 3),
]

# Swept on top of every entry above. 1 is plain row-major order; a real GEMM
# wants the L2 swizzle, and leaving it out is worth 3x at 8192**3.
SPEED_GROUP_M = [1, 8]


# A healthy sparse kernel has measured 1.3x - 1.75x of the dense one wherever the
# tensor core is the bottleneck.
SPEED_PASS = 1.30
# Two things make a sparse kernel land at or below 1.0x, and they are far apart
# in magnitude, so the threshold between them can just be the ratio:
#
#   a broken lowering  -- the bug that made Blackwell emit the legacy `mma.sp`
#                         spelling put fp16 and int8 at 0.44x and 0.23x
#   nothing to win     -- a shape whose tensor cores are idle. Halving the MMA
#                         work buys nothing and sparsity still pays for the
#                         metadata traffic, so it lands a little under 1.0x.
#                         Measured on Thor at 8192**3: int8 0.86x, fp8 0.95x,
#                         with the dense kernel within 5% of cuBLAS.
#
# So call it a failure only well below 1.0x. Between there and SPEED_PASS the
# honest answer is "this shape has no room for sparsity", which is what
# probe_mmav5_metadata.py's MMA sensitivity confirms directly.
SPEED_FAIL = 0.80
# Below this fraction of the torch kernel, the dense side is simply under-tuned
# and the ratio says nothing either way.
DENSE_TRUSTWORTHY = 0.60


def speed_verdict(sparse, dense, lib):
    """PASS / WARN / FAIL / INCONCLUSIVE for one dtype."""
    if not sparse or not dense:
        return "INCONCLUSIVE", "no timing"
    ratio = sparse / dense
    if lib and dense < DENSE_TRUSTWORTHY * lib:
        return "INCONCLUSIVE", f"dense {dense:.0f} << torch {lib:.0f}, widen SPEED_CONFIGS"
    if ratio < SPEED_FAIL:
        return "FAIL", f"{ratio:.2f}x -- sparse is far slower than dense, suspect the lowering"
    if ratio < 1.0:
        return "WARN", f"{ratio:.2f}x -- no gain here; check MMA sensitivity before blaming the lowering"
    if ratio < SPEED_PASS:
        return "WARN", f"{ratio:.2f}x -- below {SPEED_PASS:.2f}x, data-movement bound?"
    return "PASS", f"{ratio:.2f}x"


def _interleaved(launch_sparse, sp_cfgs, launch_dense, dn_cfgs, lib_fn, M, N, K, rounds=3):
    """Re-time every candidate interleaved and pick the winner from the medians.

    The sweep that produced `sp_cfgs`/`dn_cfgs` timed each config once, over a
    long enough stretch that the clock drifts across it, so a config measured at
    a lucky moment can top the sweep without being the fastest. Re-timing only
    that one config faithfully measures the wrong config -- int8 landed on 70
    TF/s in one run and 130 in the next from this alone. Carrying the top few
    through and choosing on the interleaved medians fixes the choice as well as
    the number.

    Returns (sparse TF/s, sparse cfg, dense TF/s, library TF/s).
    """
    import statistics

    def bench(fn):
        try:
            return _tf(triton.testing.do_bench(fn, warmup=25, rep=100), M, N, K)
        except Exception:  # noqa: BLE001  (no library kernel for this dtype here)
            return None

    samples = {}

    def record(key, fn):
        v = bench(fn)
        if v is not None:
            samples.setdefault(key, []).append(v)

    for _ in range(rounds):
        for cfg in sp_cfgs:
            record(("sp", cfg), lambda c=cfg: launch_sparse(*c))
        for cfg in dn_cfgs:
            record(("dn", cfg), lambda c=cfg: launch_dense(*c))
        if lib_fn:
            record(("lib", None), lib_fn)

    def best(tag):
        cands = [(statistics.median(v), k[1]) for k, v in samples.items() if k[0] == tag]
        return max(cands) if cands else (None, None)

    sp, sp_cfg = best("sp")
    dn, _ = best("dn")
    lib = statistics.median(samples[("lib", None)]) if ("lib", None) in samples else None
    return sp, sp_cfg, dn, lib


def _tf(ms, M, N, K):
    """Dense-equivalent TF/s, so sparse and dense are directly comparable."""
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def warm_up_device(device, seconds=3.0):
    """Spin the GPU up before timing anything.

    A cold consumer part sits at idle clocks, and whichever dtype happens to be
    measured first absorbs the whole ramp -- which reads as that dtype being
    slow. Burn a few seconds of dense matmul first so every dtype is measured at
    the same clocks.
    """
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            x = torch.matmul(x, x) * 0 + x
    torch.cuda.synchronize()


# How many of the sweep's leaders go on to the interleaved re-timing. The sweep
# times each config once and cannot tell a fast config from a lucky moment, so
# it nominates rather than decides.
SPEED_CANDIDATES = 4


def _best(launch, M, N, K, verbose=False, label=""):
    """Return the sweep's fastest `SPEED_CANDIDATES` configs, best-first."""
    timed = []
    for base in SPEED_CONFIGS:
        for group_m in SPEED_GROUP_M:
            cfg = (*base, group_m)
            try:
                ms = triton.testing.do_bench(lambda c=cfg: launch(*c), warmup=25, rep=100)
            except Exception as exc:  # noqa: BLE001  (a config that does not fit this device)
                if verbose:
                    print(f"      {label} {cfg}: {type(exc).__name__}")
                continue
            if verbose:
                print(f"      {label} {cfg[0]}x{cfg[1]}x{cfg[2]} w{cfg[3]} st{cfg[4]} g{cfg[5]}: "
                      f"{_tf(ms, M, N, K):7.1f} TF/s")
            timed.append((ms, cfg))
    timed.sort(key=lambda t: t[0])
    return [cfg for _, cfg in timed[:SPEED_CANDIDATES]]


def measure_speed(dtype, M, N, K, out_dtype, device, verbose=False):
    """Best-of-sweep sparse vs dense vs torch, all dense-equivalent TF/s."""
    tdt = TORCH_DTYPE[dtype]
    is_int = dtype == "int8"
    is_8bit = dtype in ("int8", "float8e4nv", "float8e5")
    acc_torch = {tl.int32: torch.int32, tl.float32: torch.float32, tl.float16: torch.float16}[out_dtype]

    if is_int:
        a = torch.randint(-4, 5, (M, K // 2), device=device, dtype=torch.int8)
        a_full = torch.randint(-4, 5, (M, K), device=device, dtype=torch.int8)
        b = torch.randint(-4, 5, (K, N), device=device, dtype=torch.int8)
    else:
        a = torch.randn(M, K // 2, device=device).to(tdt)
        a_full = torch.randn(M, K, device=device).to(tdt)
        b = torch.randn(K, N, device=device).to(tdt)
    # 8-bit operands need a K-contiguous B: the 8-bit ldmatrix cannot transpose,
    # so an N-contiguous B collapses the shared-memory path. Benchmark the
    # layout a user should actually pick.
    if is_8bit:
        b = b.T.contiguous().T
    # Any valid 2:4 pattern; correctness is the table above, not this.
    meta = torch.full((M, K // 16), 0x4444, device=device, dtype=torch.int16)
    c = torch.empty((M, N), device=device, dtype=acc_torch)

    def launch_sparse(bm, bn, bk, w, st, gm):
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        sparse_matmul_kernel[grid](a, b, c, meta, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                   c.stride(0), c.stride(1), meta.stride(0), meta.stride(1), BLOCK_M=bm, BLOCK_N=bn,
                                   BLOCK_K=bk, OUT_DTYPE=out_dtype, GROUP_M=gm, num_warps=w, num_stages=st)

    def launch_dense(bm, bn, bk, w, st, gm):
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        dense_matmul_kernel[grid](a_full, b, c, M, N, K, a_full.stride(0), a_full.stride(1), b.stride(0), b.stride(1),
                                  c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, OUT_DTYPE=out_dtype,
                                  GROUP_M=gm, num_warps=w, num_stages=st)

    lib_fn = None
    if is_int:
        lib_fn = lambda: torch._int_mm(a_full, b)  # noqa: E731
    elif is_8bit:
        one = torch.tensor(1.0, device=device)
        bt = torch.randn(N, K, device=device).to(tdt).T  # _scaled_mm wants a col-major B
        lib_fn = lambda: torch._scaled_mm(a_full, bt, scale_a=one, scale_b=one, out_dtype=torch.float32)  # noqa: E731
    else:
        lib_fn = lambda: torch.matmul(a_full, b)  # noqa: E731

    sp_cfgs = _best(launch_sparse, M, N, K, verbose, "sparse")
    dn_cfgs = _best(launch_dense, M, N, K, verbose, "dense ")

    # Re-time the candidates **interleaved**. Measuring the library kernel last
    # instead once made sparse look 1.7x faster than a dense baseline that was
    # really only 5% behind.
    sp, sp_cfg, dn, lib = _interleaved(launch_sparse, sp_cfgs, launch_dense, dn_cfgs, lib_fn, M, N, K)
    return sp, dn, lib, sp_cfg


LINEAR_RE = re.compile(r"#linear = #ttg\.linear<\{[^}]*\}>")
WARPS_RE = re.compile(r"#mma = #ttg\.nvidia_mma<\{versionMajor = (\d+)[^}]*warpsPerCTA = \[(\d+), (\d+)\]")


def metadata_layout_for(arch, dtype, shape, out_dtype):
    """The #linear the compiler attaches to the metadata, for any target."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    _, _, _, BLOCK_M, BLOCK_N, BLOCK_K, num_warps = shape
    ptr = {"float16": "*fp16", "bfloat16": "*bf16", "int8": "*i8",
           "float8e4nv": "*fp8e4nv", "float8e5": "*fp8e5"}[dtype]
    is_int = dtype == "int8"
    sig = {"a_ptr": ptr, "b_ptr": ptr, "c_ptr": "*i32" if is_int else "*fp32", "meta_ptr": "*i16",
           "M": "i32", "N": "i32", "K": "i32",
           "stride_am": "i32", "stride_ak": "constexpr", "stride_bk": "i32", "stride_bn": "constexpr",
           "stride_cm": "i32", "stride_cn": "constexpr", "stride_mm": "i32", "stride_mk": "constexpr",
           "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "OUT_DTYPE": "constexpr"}
    ce = {"stride_ak": 1, "stride_bn": 1, "stride_cn": 1, "stride_mk": 1,
          "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "OUT_DTYPE": out_dtype}
    src = ASTSource(fn=sparse_matmul_kernel, signature=sig, constexprs=ce)
    ttgir = triton.compile(src, target=GPUTarget("cuda", arch, 32),
                           options={"num_warps": num_warps, "num_stages": 3}).asm["ttgir"]
    lin = LINEAR_RE.search(ttgir)
    w = WARPS_RE.search(ttgir)
    return (lin.group(0) if lin else None,
            (int(w.group(1)), int(w.group(2)), int(w.group(3))) if w else None)


def report_v3_coverage(dtypes, shapes, cc):
    """How much of the Hopper metadata layout this run just covered.

    MMAv2 and MMAv3 read the metadata through the same in-warp mapping -- the PTX
    ISA documents both with the same figures -- and differ only in warp-basis
    order. Where warpsPerCTA[1] == 1 there are no N-warps to broadcast over and
    the two layouts coincide exactly, so a bit-exact pass here is a bit-exact
    pass for the layout Hopper would use. Nothing else about MMAv3 (the shared
    memory descriptors, the K accounting, wgmma itself) transfers.
    """
    print("\nsm_90 (MMAv3) metadata-layout coverage from this run")
    print(f"{'dtype':11s} {'block':14s} {'w':>2s} {'local':>10s} {'sm_90':>10s}  verdict")
    print("-" * 76)
    covered = reachable = 0
    for dtype in dtypes:
        out_dtype = tl.int32 if dtype == "int8" else tl.float32
        seen = set()
        for shape in shapes:
            key = (dtype, shape[3:7])
            if key in seen:
                continue
            seen.add(key)
            try:
                l_loc, w_loc = metadata_layout_for(cc, dtype, shape, out_dtype)
                l_90, w_90 = metadata_layout_for(90, dtype, shape, out_dtype)
            except Exception:  # noqa: BLE001
                continue
            if l_loc is None or l_90 is None or w_90 is None:
                continue
            if w_90[0] != 3:
                # sm_90 cannot use wgmma here (M or num_warps too small) and
                # falls back to MMAv2, so this shape says nothing about MMAv3.
                verdict = "n/a -- sm_90 falls back to MMAv2"
            else:
                reachable += 1
                if l_loc == l_90:
                    covered += 1
                    verdict = "COVERED -- identical layout, verified above"
                else:
                    verdict = "needs Hopper (N-warps reorder the warp bases)"
            blk = "x".join(map(str, shape[3:6]))
            wl = f"v{w_loc[0]}[{w_loc[1]},{w_loc[2]}]" if w_loc else "?"
            w9 = f"v{w_90[0]}[{w_90[1]},{w_90[2]}]"
            print(f"{dtype:11s} {blk:14s} {shape[6]:>2d} {wl:>10s} {w9:>10s}  {verdict}")
    if reachable:
        print(f"\n{covered}/{reachable} of the shapes that actually reach MMAv3 use a metadata")
        print("layout byte-identical to this chip's, so the bit-exact table above covers")
        print("them on real silicon. The rest differ only in where the broadcast N-warp")
        print("bases sit among the warp bits, and need a Hopper device -- as does")
        print("everything else about MMAv3 (shared memory descriptors, K accounting, wgmma).")
    else:
        print("\nNo shape here reaches MMAv3 (needs M >= 64 and num_warps % 4 == 0).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default=None, help="comma-separated subset to test")
    ap.add_argument("--quick", action="store_true", help="one shape per dtype")
    ap.add_argument("--speed-size", type=int, default=2048, help="square size for the speed table (0 to skip)")
    ap.add_argument("--speed-verbose", action="store_true", help="print every config in the speed sweep")
    ap.add_argument("--no-warmup", action="store_true", help="skip the clock warm-up (not recommended)")
    ap.add_argument("--no-v3-coverage", action="store_true",
                    help="skip the sm_90 metadata-layout coverage report")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    name = torch.cuda.get_device_name()
    print(f"device : {name} (sm_{cc})")
    print(f"triton : {triton.__version__}")
    print(f"torch  : {torch.__version__}\n")

    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_supported_sparse_dot_dtypes
    supported_fn = get_supported_sparse_dot_dtypes(GPUTarget("cuda", cc, 32))

    candidates = ["float16", "bfloat16", "int8", "float8e4nv", "float8e5"]
    if args.dtypes:
        candidates = args.dtypes.split(",")
    dtypes = []
    for d in candidates:
        # get_supported_sparse_dot_dtypes keys off the tl dtype's own name, so
        # ask it with the tl dtype rather than reinventing the naming.
        if supported_fn(getattr(tl, d)):
            dtypes.append(d)
        else:
            print(f"skip   : {d} (unsupported on sm_{cc})")
    if not dtypes and 90 <= cc < 100:
        print("\nsm_90's MMAv3 sparse path ships disabled: its numerics have never run")
        print("on a Hopper device. This script is how that gets closed -- re-run with")
        print("    TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1 python verify_dot_sparse.py")
        print("and report the verdict block. Nothing else needs changing.")
        return 2
    if dtypes:
        print()

    counters = {"pass": 0, "fail": 0, "skip": 0}
    instrs = {}
    verdicts = {}

    def report(label, dtype, shape, out_dtype):
        try:
            ok, detail, instr = run_case(dtype, shape, out_dtype, "cuda")
            state = "PASS" if ok else "FAIL"
        except triton.runtime.errors.OutOfResources as exc:
            # A shared-memory budget this device does not have is a property of
            # the device, not of the sparse path.
            state, detail, instr = "SKIP", f"{type(exc).__name__} (shared memory)", "-"
        except Exception as exc:  # noqa: BLE001
            state, detail, instr = "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}", "-"
        counters[state.lower()] += 1
        if instr != "-":
            instrs[instr] = instrs.get(instr, 0) + 1
        mnk = "x".join(map(str, shape[:3]))
        blk = "x".join(map(str, shape[3:6]))
        print(f"[{state}] {label:11s} {mnk:14s} block {blk:14s} w{shape[6]}  {detail:26s} {instr}")

    for dtype in dtypes:
        out_dtype = tl.int32 if dtype == "int8" else tl.float32
        shapes = SHAPES[-1:] if args.quick else SHAPES
        if not args.quick and dtype in ("int8", "float8e4nv", "float8e5"):
            shapes = shapes + EIGHT_BIT_SHAPES
        for shape in shapes:
            report(dtype, dtype, shape, out_dtype)

    # The fp16 accumulator is an opt-in variant that only exists for fp16 inputs.
    if "float16" in dtypes:
        print()
        for shape in (SHAPES[-1:] if args.quick else SHAPES):
            report("f16-acc", "float16", shape, tl.float16)

    print("\ninstructions actually exercised:")
    for instr, n in sorted(instrs.items()):
        print(f"  {n:3d}x  {instr}")
    print(f"\n{'FAILED' if counters['fail'] else 'ALL PASSED'} "
          f"({counters['pass']} passed, {counters['fail']} failed, {counters['skip']} skipped)")

    if not args.no_v3_coverage and dtypes and not counters["fail"]:
        report_v3_coverage(dtypes, SHAPES[-1:] if args.quick else SHAPES, cc)

    # Correctness says the layout is right; it says nothing about speed, which is
    # the whole point of 2:4 sparsity. Same GEMM, same config sweep, both paths.
    if args.speed_size and dtypes:
        S = args.speed_size
        if not args.no_warmup:
            print("\nwarming the device up (idle clocks would penalise whichever "
                  "dtype is measured first)...", flush=True)
            warm_up_device("cuda")
        print(f"speed  {S}x{S}x{S}, dense-equivalent TF/s "
              f"(8-bit uses a K-contiguous B, as a user should)")
        print(f"{'dtype':13s} {'sparse':>8s} {'dense':>8s} {'torch':>8s}  "
              f"{'verdict':<12s} detail")
        print("-" * 88)
        speedups = []
        verdicts = {}
        for dtype in dtypes:
            out_dtype = tl.int32 if dtype == "int8" else tl.float32
            try:
                sp, dn, lib, cfg = measure_speed(dtype, S, S, S, out_dtype, "cuda", args.speed_verbose)
            except Exception as exc:  # noqa: BLE001
                print(f"{dtype:13s} {type(exc).__name__}: {str(exc).splitlines()[0][:50]}")
                continue
            if sp and dn:
                speedups.append(sp / dn)
            num = lambda v: f"{v:8.1f}" if v else "       -"  # noqa: E731
            state, detail = speed_verdict(sp, dn, lib)
            verdicts[dtype] = state
            print(f"{dtype:13s} {num(sp)} {num(dn)} {num(lib)}  {state:<12s} {detail}")
            if dtype == "float16":
                sp16, _, _, cfg16 = measure_speed(dtype, S, S, S, tl.float16, "cuda", args.speed_verbose)
                gain = f"{sp16 / sp:.2f}x vs sparse f32 acc" if (sp16 and sp) else "-"
                print(f"{'  f16 acc':13s} {num(sp16)} {'':>8s} {'':>8s}  {'':<12s} {gain}")
        if speedups:
            print(f"\nsparse/dense spread: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print(f"PASS >= {SPEED_PASS:.2f}x, WARN {SPEED_FAIL:.2f}-{SPEED_PASS:.2f}x, "
              f"FAIL < {SPEED_FAIL:.2f}x (sparse slower than dense).")
        print("Sparsity only pays where the tensor cores are the bottleneck. On a")
        print("bandwidth-bound shape halving the MMA work buys nothing, and this table")
        print("reads 1.0x through no fault of the lowering -- on Jetson Thor fp16 sits")
        print("at 1.05x of dense at 4096**3 and reaches 1.50x at 8192**3. So raise")
        print("--speed-size before concluding anything from a low ratio.")
        print("probe_mmav5_metadata.py separates the two: its MMA-sensitivity number is")
        print("how much the runtime moves when the MMA count halves. Near 1.00x means")
        print("the shape has no room for sparsity and this row says nothing about the")
        print("lowering. A ratio below 1.0 with a sensitivity well above 1.0 is the")
        print("signature of a real defect, e.g. the legacy mma.sp spelling on Blackwell.")

    # ------------------------------------------------------------- verdict
    n_fail = sum(v == "FAIL" for v in verdicts.values())
    n_warn = sum(v == "WARN" for v in verdicts.values())
    n_inc = sum(v == "INCONCLUSIVE" for v in verdicts.values())
    n_pass = sum(v == "PASS" for v in verdicts.values())
    correctness_ok = counters["fail"] == 0
    speed_ok = n_fail == 0

    bar = "=" * 60
    print(f"\n{bar}")
    c_state = "PASS" if correctness_ok else "FAIL"
    print(f"  correctness : {c_state}  "
          f"({counters['pass']} passed, {counters['fail']} failed, {counters['skip']} skipped)")
    if verdicts:
        s_state = "PASS" if speed_ok else "FAIL"
        bits = [f"{n_pass} pass"]
        if n_warn:
            bits.append(f"{n_warn} warn")
        if n_inc:
            bits.append(f"{n_inc} inconclusive")
        if n_fail:
            bits.append(f"{n_fail} FAIL")
        print(f"  speed       : {s_state}  ({', '.join(bits)})")
        for d, v in verdicts.items():
            if v != "PASS":
                print(f"                  {v}: {d}")
    else:
        print("  speed       : not measured (--speed-size 0)")
    overall = "ALL GOOD" if (correctness_ok and speed_ok) else "PROBLEM FOUND"
    print(f"  => {overall}")
    print(bar)

    return 0 if (correctness_ok and speed_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
