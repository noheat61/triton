#!/usr/bin/env python3
"""How much of a tcgen05.mma.sp kernel's time is the metadata TMEM round trip?

`tcgen05.mma.sp` wants its sparsity metadata in tensor memory, so the compiler
emits, once per K iteration, a `tcgen05.st` of the metadata followed by a
`tcgen05.wait::st` barrier before the MMA. A and B do not pay this: the pipeliner
gives them async copies into a multi-buffered shared-memory staging buffer, so
their loads overlap the MMA. The metadata gets a single TMEM buffer and a
synchronous store, which serialises every iteration.

That is a property of the lowering, not of any one chip -- it is the same on
sm_100 -- so it is worth measuring rather than inferring. This script measures it
by ablation:

    sparse            metadata advances every iteration -> store + barrier in the loop
    sparse-static     metadata pointer never advances, so LICM hoists the load and
                      the TMEM store out of the loop

Both run the identical number of `tcgen05.mma.sp` instructions over identical A
and B traffic. `sparse-static` computes a wrong answer -- every K tile reuses one
metadata tile -- but it is the right *timing* control: the only thing removed is
the per-iteration metadata path. The gap between them is its cost.

Also reported, because they are cheap once the harness exists:

    dense             tcgen05.mma, the thing sparsity has to beat
    sparse (M=64)     the same dot routed to mma.sp (MMAv2), whose metadata is a
                      register operand and needs no TMEM at all

Run it on a datacenter Blackwell part (sm_100/103/110). On anything older every
row lands on mma.sp and the ablation is vacuous -- the script says so.
"""
import argparse
import re
import sys

import torch
import triton
import triton.language as tl

SPARSE_INSTR = re.compile(r"(?:tcgen05\.mma\.sp|wgmma\.mma_async\.sp|mma\.sp)"
                          r"(?:::ordered_metadata)?[\w:.]*")
DENSE_INSTR = re.compile(r"(?:tcgen05\.mma(?!\.sp)|wgmma\.mma_async(?!\.sp)|mma\.sync)[\w:.]*")


@triton.jit
def sparse_kernel(a_ptr, b_ptr, c_ptr, meta_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_mm, stride_mk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, OUT_DTYPE: tl.constexpr, ADVANCE_META: tl.constexpr,
                  GROUP_M: tl.constexpr = 1):
    # L2 CTA swizzle. Without it both kernels lose a factor of three at 8192**3,
    # which is larger than anything this script is trying to measure. GROUP_M=1
    # is the plain row-major order.
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
        # constexpr, so with ADVANCE_META=False m_ptrs is loop-invariant and the
        # metadata load -- and the TMEM store the MMA lowering hangs off it --
        # leave the loop.
        if ADVANCE_META:
            m_ptrs += (BLOCK_K // 16) * stride_mk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def dense_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 OUT_DTYPE: tl.constexpr, GROUP_M: tl.constexpr = 1):
    # L2 CTA swizzle. Without it both kernels lose a factor of three at 8192**3,
    # which is larger than anything this script is trying to measure. GROUP_M=1
    # is the plain row-major order.
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


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS_128 = [
    (128, 128, 64, 4, 3),
    (128, 128, 128, 4, 3),
    (128, 128, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (128, 256, 128, 8, 3),
    (256, 128, 64, 8, 3),
]
# The MMAv2 comparison point: M < 128 is what routes a datacenter-Blackwell
# 8-bit sparse dot away from tcgen05 and onto mma.sp::ordered_metadata.
CONFIGS_64 = [
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
    (64, 256, 128, 8, 3),
]
# M = 32 is not a multiple of 64, so supportMMA(v5) rejects it and the *dense*
# dot falls back to mma.sync too. That makes the two M=32 rows a like-for-like
# legacy-MMA comparison, which is the only way to tell "the sparse fallback is
# slow" apart from "the whole legacy warp-level MMA path is slow for this dtype
# on this part".
CONFIGS_32 = [
    (32, 128, 64, 4, 3),
    (32, 128, 128, 4, 3),
    (32, 256, 128, 4, 3),
]


def census(compiled):
    """(sparse instr, dense instr, #tcgen05.st, #wait::st, metadata TMEM depth)."""
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm["ttgir"]
    sp = SPARSE_INSTR.search(ptx)
    dn = DENSE_INSTR.search(ptx)
    meta = re.findall(r"memdesc<([0-9x]+)i16, #tmem", ttgir)
    return {
        "instr": (sp or dn).group(0) if (sp or dn) else "?",
        "n_mma": len(re.findall(r"tcgen05\.mma[\w:.]*|mma\.sp[\w:.]*|mma\.sync[\w:.]*", ptx)),
        "n_st": len(re.findall(r"tcgen05\.st\.", ptx)),
        "n_wait_st": len(re.findall(r"tcgen05\.wait::st", ptx)),
        "meta_buf": meta[0] if meta else "-- (registers, no TMEM)",
    }


def tf(ms, M, N, K):
    """Dense-equivalent TF/s, so every row is directly comparable."""
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def warm_up(device, seconds=3.0):
    import time
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            x = torch.matmul(x, x) * 0 + x
    torch.cuda.synchronize()


# Swept on top of every entry in CONFIGS_*; 1 is plain row-major order.
GROUP_MS = [1, 8]


def best(launch, configs, M, N, K, verbose):
    best_ms, best_cfg, best_k = float("inf"), None, None
    for base in configs:
        for group_m in GROUP_MS:
            cfg = (*base, group_m)
            try:
                k = launch(*cfg)
                ms = triton.testing.do_bench(lambda c=cfg: launch(*c), warmup=25, rep=100)
            except Exception as exc:  # noqa: BLE001  (a config this device cannot fit)
                if verbose:
                    print(f"      {cfg}: {type(exc).__name__}: {exc}"[:150])
                continue
            if verbose:
                print(f"      {cfg[0]}x{cfg[1]}x{cfg[2]} w{cfg[3]} st{cfg[4]} g{cfg[5]}: "
                      f"{tf(ms, M, N, K):7.1f} TF/s")
            if ms < best_ms:
                best_ms, best_cfg, best_k = ms, cfg, k
    if best_cfg is None:
        return None, None, None
    return tf(best_ms, M, N, K), best_cfg, best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048, help="square M=N=K")
    ap.add_argument("--dtype", default="float8e4nv",
                    choices=["float8e4nv", "float8e5", "int8", "float16", "bfloat16"])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    S = args.size
    M = N = K = S
    print(f"device : {torch.cuda.get_device_name()} (sm_{cc})")
    print(f"triton : {triton.__version__}")
    print(f"shape  : {M}x{N}x{K}  dtype {args.dtype}\n")

    is_int = args.dtype == "int8"
    tdt = {
        "float8e4nv": torch.float8_e4m3fn, "float8e5": torch.float8_e5m2, "int8": torch.int8,
        "float16": torch.float16, "bfloat16": torch.bfloat16,
    }[args.dtype]
    is_16bit = args.dtype in ("float16", "bfloat16")
    out_dtype = tl.int32 if is_int else tl.float32
    acc_torch = torch.int32 if is_int else torch.float32

    dev = "cuda"
    if is_int:
        a = torch.randint(-4, 5, (M, K // 2), device=dev, dtype=torch.int8)
        a_full = torch.randint(-4, 5, (M, K), device=dev, dtype=torch.int8)
        b = torch.randint(-4, 5, (K, N), device=dev, dtype=torch.int8)
    else:
        a = torch.randn(M, K // 2, device=dev).to(tdt)
        a_full = torch.randn(M, K, device=dev).to(tdt)
        b = torch.randn(K, N, device=dev).to(tdt)
    if not is_16bit:
        # 8-bit ldmatrix cannot transpose; give it a K-contiguous B. 16-bit can,
        # so leave B in the layout torch.matmul would hand it.
        b = b.T.contiguous().T
    meta = torch.full((M, K // 16), 0x4444, device=dev, dtype=torch.int16)
    c = torch.empty((M, N), device=dev, dtype=acc_torch)

    def sparse_launch(advance):

        def go(bm, bn, bk, w, st, gm):
            grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
            return sparse_kernel[grid](a, b, c, meta, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                       c.stride(0), c.stride(1), meta.stride(0), meta.stride(1), BLOCK_M=bm,
                                       BLOCK_N=bn, BLOCK_K=bk, OUT_DTYPE=out_dtype, ADVANCE_META=advance,
                                       GROUP_M=gm, num_warps=w, num_stages=st)

        return go

    def dense_launch(bm, bn, bk, w, st, gm):
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        return dense_kernel[grid](a_full, b, c, M, N, K, a_full.stride(0), a_full.stride(1), b.stride(0), b.stride(1),
                                  c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, OUT_DTYPE=out_dtype,
                                  GROUP_M=gm, num_warps=w, num_stages=st)

    if not args.no_warmup:
        print("warming the device up...")
        warm_up(dev)

    rows = []
    for label, launch, cfgs in (
        ("dense", dense_launch, CONFIGS_128),
        ("sparse", sparse_launch(True), CONFIGS_128),
        ("sparse-static-meta", sparse_launch(False), CONFIGS_128),
        ("sparse M=64 (mma.sp)", sparse_launch(True), CONFIGS_64),
        ("dense  M=32 (mma.sync)", dense_launch, CONFIGS_32),
        ("sparse M=32 (mma.sp)", sparse_launch(True), CONFIGS_32),
    ):
        if args.verbose:
            print(f"  {label}:")
        perf, cfg, k = best(launch, cfgs, M, N, K, args.verbose)
        rows.append((label, perf, cfg, census(k) if k else None))

    print(f"\n{'variant':22s} {'TF/s':>7s}  {'block':>14s}  {'mma':>4s} {'st':>3s} {'wait':>4s}  "
          f"{'meta TMEM buf':16s} instruction")
    print("-" * 118)
    for label, perf, cfg, cen in rows:
        if perf is None:
            print(f"{label:22s} {'--':>7s}  (no config compiled on this device)")
            continue
        blk = f"{cfg[0]}x{cfg[1]}x{cfg[2]}w{cfg[3]}"
        print(f"{label:22s} {perf:7.1f}  {blk:>14s}  {cen['n_mma']:4d} {cen['n_st']:3d} {cen['n_wait_st']:4d}  "
              f"{cen['meta_buf']:16s} {cen['instr']}")

    d = dict((r[0], r[1]) for r in rows)
    cen = dict((r[0], r[3]) for r in rows)
    sp, st_, dn, v2 = (d["sparse"], d["sparse-static-meta"], d["dense"], d["sparse M=64 (mma.sp)"])
    print()

    if not cen.get("sparse") or "tcgen05" not in cen["sparse"]["instr"]:
        print("This device does not route the sparse dot through tcgen05.mma.sp, so the")
        print("metadata never reaches tensor memory. Run on sm_100 / sm_103 / sm_110.")
        return 0

    # Did the ablation actually remove anything? LICM has to hoist the metadata
    # load AND the TMEM store the MMA lowering hangs off it. TMEMAllocOp is not
    # hoistable on its own, so this often does not happen -- in which case the
    # control differs from `sparse` only in that it re-reads one metadata tile
    # from cache, and says nothing about the TMEM store.
    ablated = (cen["sparse-static-meta"]["n_st"] < cen["sparse"]["n_st"]
               or cen["sparse-static-meta"]["n_wait_st"] < cen["sparse"]["n_wait_st"])

    print(f"sparse vs dense               : {sp / dn:5.2f}x   <- what a user gets today")
    print(f"sparse-static-meta vs dense   : {st_ / dn:5.2f}x")
    if v2:
        print(f"mma.sp (M=64) vs dense        : {v2 / dn:5.2f}x   <- MMAv2 route, metadata in registers")
    dn32, sp32 = d.get("dense  M=32 (mma.sync)"), d.get("sparse M=32 (mma.sp)")
    if dn32 and sp32:
        print(f"legacy MMA, like for like     : dense(M=32) {dn32:.1f} vs sparse(M=32) {sp32:.1f} "
              f"= {sp32 / dn32:5.2f}x")
        print(f"legacy vs tcgen05, dense only : {dn32 / dn:5.2f}x   <- 1.0 means the legacy path is "
              f"competitive here")
        if dn32 / dn < 0.25:
            print()
            print(f"  NOTE: the legacy warp-level MMA is {dn / dn32:.0f}x slower than tcgen05 for this dtype")
            print("  on this part, for the DENSE dot too. That is a property of the instruction,")
            print("  not of sparsity -- so routing a sparse dot to mma.sp here is a trap even")
            print("  though it is bit-exact. Compare the same row with --dtype int8.")
    print()

    if not ablated:
        print("ABLATION INEFFECTIVE: sparse-static-meta still emits the same number of")
        print(f"  tcgen05.st ({cen['sparse']['n_st']}) and tcgen05.wait::st ({cen['sparse']['n_wait_st']}) as sparse, so the metadata")
        print("  TMEM store was NOT hoisted out of the loop -- LICM will not move a")
        print("  TMEMAllocOp. The gap between the two rows is only the cost of re-reading")
        print(f"  the metadata from global memory ({(st_ / sp - 1) * 100:.0f}%), not the TMEM round trip.")
        print("  Nothing here measures the store+barrier; use ncu for that.")
        print()

    # The part that does not depend on the ablation: sparse issues half the MMA
    # instructions of dense and reads half the A bytes. If neither buys time, the
    # tensor core is not what the kernel is waiting on, and no sparse instruction
    # -- however well pipelined -- can help at this shape.
    n_sp, n_dn = cen["sparse"]["n_mma"], cen["dense"]["n_mma"]
    best_case = max(sp, st_) / dn
    print(f"MMA instructions              : dense {n_dn}, sparse {n_sp}")
    if best_case < 1.10:
        print(f"MMA sensitivity               : {best_case:5.2f}x")
        print()
        print("VERDICT: halving the MMA count (and the A-operand bytes with it) changes")
        print("         the runtime by less than 10%. The tensor core is not the")
        print("         bottleneck at this shape, so sparsity has no lever here and this")
        print("         measurement cannot tell you whether tcgen05.mma.sp is implemented")
        print("         well. It is a property of the shape and the part, not of the")
        print("         sparse path -- the same kernel shows 1.5-1.6x on sm_89.")
        print()
        print("         To find what it IS waiting on:")
        print("           ncu --set full --kernel-name regex:'sparse_kernel|dense_kernel' \\")
        print("               --launch-count 2 -o probe python probe_mmav5_metadata.py --size 2048")
        print("           ncu --import probe.ncu-rep --csv --page raw \\")
        print("               --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\\")
        print("dram__throughput.avg.pct_of_peak_sustained_elapsed,\\")
        print("lts__throughput.avg.pct_of_peak_sustained_elapsed")
        print("         A tensor-pipe figure far below the memory figures confirms it.")
    else:
        print(f"MMA sensitivity               : {best_case:5.2f}x  -- the tensor core does matter here.")
        print()
        print("VERDICT: this shape has headroom, so the sparse/dense ratio is meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
