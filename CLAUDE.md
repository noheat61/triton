# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [triton-lang/triton](https://github.com/triton-lang/triton) on branch `dot-sparse-nvidia`,
adding `tl.dot_sparse` (2:4 structured sparsity) to the NVIDIA backend across three tensor-core
generations. Upstream base is `087e97245`; the feature is the 17 commits prefixed `[NVIDIA]`
(`git log --grep="^\[NVIDIA\]"`). Not yet submitted upstream.

Read `AGENTS.md` first — it carries the upstream rules (block programming model, descriptor memory
effects, C++/lowering guidelines) and applies to all work here.

Three project docs, all in Korean:

| File | Contents |
|---|---|
| `TODO.md` | Per-generation status, settled design decisions, measured perf, **"재시도 금지"** (dead ends — read before re-attempting anything) |
| `DOT_SPARSE_INTERNALS.md` | Function-level walkthrough of the whole lowering path, with file:line pointers |
| `ENV_SETUP.md` | Build environment, cross-machine wheel workflow, pitfalls table |

Keep all three current when the feature changes — they are the handoff state, not archive.

## Build and test

Build dir on this machine: `build/cmake.linux-aarch64-cpython-3.10`
(canonically `PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'`).

```bash
make                                  # incremental ninja; REQUIRED after any C++/tablegen change
                                      # skip it for Python-only edits
ninja -C build/cmake.linux-aarch64-cpython-3.10 triton-opt   # lit tests only need this target
```

`ninja` is memory-hungry: one TU is ~2 GB, so cap jobs (`MAX_JOBS=3` on the 15 GB x86 boxes; this
Thor box has more headroom). Never pipe `ninja` into `head`/`grep` — SIGPIPE can kill it right
before the link, leaving fresh `.o` files behind a stale `.so`. Redirect to a file and grep after.

Tests (prefix everything with `env -u PYTHONPATH` if a system `PYTHONPATH` such as ROS is set):

```bash
# lit — no GPU needed, and cross-architecture
cd build/cmake.linux-aarch64-cpython-3.10 && lit -v test/TritonGPU/accelerate-matmul.mlir
make test-lit                                      # all lit tests

# pytest
pytest -s --tb=short python/test/unit/language/test_core.py -k dot_sparse       # numerics (needs GPU)
pytest -s python/test/unit/language/test_compile_errors.py -k dot_sparse        # frontend rejections
pytest -s python/test/unit/language/test_compile_only.py -k dot_sparse          # cross-target, no GPU
pytest python/test/unit/language/test_core.py::test_dot_sparse                  # single test
```

### Verifying without the target GPU

`triton.compile(ASTSource(...), target=GPUTarget("cuda", cc, 32))` runs TTIR→TTGIR→LLVM→PTX→**cubin**
for an architecture the machine does not have (ptxas comes from `~/.triton/nvidia/`). That covers
everything except numerics, and is how sm_90/sm_100/sm_120 support is regression-tested — see
`_compile_sparse_matmul` in `python/test/unit/language/test_compile_only.py`, which pins which sparse
instruction each capability selects. When only PTX legality is in question, assembling one line with
ptxas directly (varying `.target`) is faster still.

Numerics on a machine without the repo: `dot-sparse-scripts/verify_dot_sparse.py` needs only a
triton wheel plus torch, reports bit-exactness per dtype/shape and which instruction was selected
(so a silent fallback to an older MMA version is visible). `dot-sparse-scripts/setup_target_conda.sh`
provisions a fresh GPU box for it.

Two more scripts back the performance claims, and both exist because the naive measurement is
misleading: `probe_mmav5_metadata.py` reports **MMA sensitivity** — how much the runtime moves when
the MMA count halves — which is what separates "the lowering is wrong" from "this shape leaves the
tensor cores idle, so sparsity has no lever"; `bench_tma_sparse.py` runs the TMA + warp-specialized
persistent path. Every one of them sweeps the L2 CTA swizzle and re-times the winners interleaved,
because on this class of device clock drift between sequential measurements is larger than the
effect being measured.

**This machine is a Jetson Thor (sm_110, aarch64, Python 3.10)** — Blackwell with TMEM/tcgen05. It
is what validated the MMAv5 sparse path (`kind::f16`, `kind::f8f6f4` and `kind::i8`, 54/54
bit-exact). It cannot stand in for sm_90, where `WGMMA.cpp` is never executed.

Do not take a target-feature gate on faith when ptxas can be asked directly: upstream's
`supportsI8Tcgen05MMA()` read `cc == 100`, but ptxas assembles `tcgen05.mma.kind::i8` for sm_110a
and rejects it for sm_103a by name, and opening the gate for Thor doubled the dense int8 kernel
while keeping results exact.

Building needs `MAX_JOBS`/`-j` set explicitly — see the build section. `ninja`, `cmake`, `nanobind`,
`lit`, and `pytest` live in the `triton_sparse` conda env; the build dir was originally configured
by a pip isolated build whose toolchain is gone, so reconfigure with the system `cmake` (3.28) and
`-DCMAKE_MAKE_PROGRAM=$CONDA_PREFIX/bin/ninja` if `build.ninja` tries to regenerate itself.

## Architecture of the sparse path

One `tl.dot_sparse(a, b, meta, acc)` reaches one PTX instruction through six stages, each owning a
disjoint concern:

| Stage | File |
|---|---|
| Builtin | `python/triton/language/core.py` |
| Semantic (shape/dtype checks, result type) | `python/triton/language/semantic.py` |
| TTIR op + verifier | `include/triton/Dialect/Triton/IR/TritonOps.td`, `lib/Dialect/Triton/IR/Ops.cpp` |
| Layout selection (TTGIR) | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` |
| Lowering to PTX | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/{MMAv2,WGMMA,MMAv5}.cpp` |
| Interpreter (`TRITON_INTERPRET=1`) | `python/triton/runtime/interpreter.py` |

Dispatch happens in `AccelerateMatmul.cpp`:

```
tt.dot_sparse
 ├─ SparseBlockedToMMA     sm_80–89, sm_120+ → tt.dot_sparse(#mma v2)      → mma.sp.sync
 │                         sm_90             → ttng.warp_group_dot + meta  → wgmma.mma_async.sp
 └─ SparseBlockedToMMAv5   sm_100–103 (8-bit)→ ttng.tc_gen5_mma  + meta    → tcgen05.mma.sp
```

Target gating lives in `third_party/nvidia/backend/compiler.py`
(`get_supported_sparse_dot_dtypes`, `get_min_sparse_dot_size`) — it is both the dtype gate and the
architecture gate, which is why the frontend error says "on this target".

### Invariants to preserve

- **The user-facing metadata contract is architecture-independent**: `meta` is i16 of shape
  `[..., M, K_dense/16]`, each i16 holding four nibbles of two 2-bit kept-element indices. The same
  kernel source runs unchanged on sm_86/90/100/120; all per-architecture difference (v2 registers,
  v3 M-first warp basis, v5 TMEM residency) is hidden in the compiler. `verifySparseDotMetadata()`
  is that contract, shared by `DotSparseOp`, `WarpGroupDotOp`, and `TCGen5MMAOp`.
- **No new IR op, and no new attribute.** Metadata layouts reuse `LinearEncodingAttr` (v2/v3) and
  `tensor_memory_encoding` (v5); v3/v5 attach an *optional operand* to the existing
  `WarpGroupDotOp`/`TCGen5MMAOp` rather than introducing ops — which is why the pipeliner, barrier,
  and accumulator passes needed no changes. A `SparseMetadataEncodingAttr` was tried and removed.
  The one addition is a defaulted `sparseMetaRowPaired` flag on `TensorMemoryEncodingAttr`, in the
  style of its existing `fp4Padded`: the `.kind::f16` metadata layout is the 8-bit one with row
  bit 3 and column bit 0 exchanged, so a flag plus one bit swap in `tensorMemoryToLinearLayout`
  covers it.
- **Metadata bit mappings are measured, not inferred.** The 8-bit k64 mapping was recovered by
  one-hot probing (first guess was entirely wrong); v3 was settled by confirming the PTX ISA reuses
  the *same figure files* as `mma.sp`; v5's layouts were read off PTX Figures 263/267 and then
  confirmed bit-exact on sm_110. The same discipline caught the metadata *addressing*: the
  `.kind::f16` metadata address names a 64-bit granule and the descriptor's sparsity selector picks
  the 32-bit half, which only showed up as a `misaligned address` fault on hardware. Do not
  extrapolate a mapping from a neighbouring one.
- **The Hopper (MMAv3) path ships disabled**, behind `knobs.nvidia.enable_unverified_sparse_wgmma`
  (`TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1`), because its numerics have never run on a Hopper
  device. The knob is part of the backend hash — any similar knob must be, or kernels compiled with
  it on keep coming out of the cache with it off.
- A shape a pattern declines must land somewhere. `SparseBlockedToMMAv5` takes M % 64 == 0 on 4 or 8
  warps, mirroring the dense rules in `supportMMA`; everything else is picked up by
  `SparseBlockedToMMA` on `mma.sp`, which covers every M >= 16. The two patterns key off one shared
  predicate so that cannot drift apart — a dot neither claims is left unlowered and fails to compile.
- M = 64 is tcgen05's Layout F: it drives only half the tensor-memory datapath lanes, and the spec
  requires A, D and the sparsity metadata to sit in the same half. `TensorMemoryAllocation` enforces
  that by joining the metadata allocation to the accumulator's row group; without the join the
  metadata lands in the other half and the instruction faults at run time.

### Performance facts that shape the code

- `mma.sp::ordered_metadata` on sm_120 is 4–6x faster than plain `mma.sp` for fp16 and int8 (no
  change for bf16/fp8) — the ptxas advisory was literal. sm_80–89 keeps legacy `mma.sp` (zero
  measured gain, and adopting it would raise the PTX floor).
- 8-bit operands must be K-contiguous: `ldmatrix.trans` transposes in 16-bit units only, so an
  N-contiguous B collapses the SMEM→register path (1.32–1.70x). This is an upstream dense limitation
  too, not a sparse-path bug, and there is no 8-bit transposing `ldmatrix` usable here.
- `out_dtype=tl.float16` selects an f16 accumulator for fp16 inputs only (1.4x on consumer chips,
  which halve f32-accumulate MMA throughput); error grows as √K.
