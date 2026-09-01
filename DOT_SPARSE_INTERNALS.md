# `tl.dot_sparse` 구현 해설

2:4 구조적 희소성을 세 세대의 텐서코어 명령으로 내리는 경로의 핵심 함수 해설.

| 세대 | 명령 | 아키텍처 | metadata 위치 |
|---|---|---|---|
| MMAv2 | `mma.sp.sync.aligned.m16n8k32` / `.m16n8k64` | sm_80–89, sm_120+ | 레지스터 |
| MMAv3 | `wgmma.mma_async.sp.sync.aligned.m64nNk32` / `k64` | sm_90 (**기본 비활성**) | 레지스터 |
| MMAv5 | `tcgen05.mma.sp.kind::{i8,f8f6f4,f16}` | sm_100–110 | **TMEM** |

프론트엔드 검증부터 PTX 생성까지, 그리고 왜 전용 IR attribute 도 전용 op 도 필요하지 않았는지.

브랜치 `dot-sparse-nvidia`, base `087e97245` (upstream/main):

```
<신규> Implement dot_sparse on Blackwell via tcgen05.mma.sp (MMAv5, 8-bit)
702490b2e Implement dot_sparse on Hopper via wgmma.mma_async.sp (MMAv3)
cecf4e99d Enable dot_sparse on consumer Blackwell (sm_120)
45c73c66d Allow an fp16 accumulator for fp16 dot_sparse
31bb6aaf5 Add 8-bit (int8/fp8) dot_sparse via mma.sp m16n8k64
f442a8934 Tidy up dot_sparse tests and tutorial
9f73d12b4 Implement dot_sparse in interpreter mode
0c47f066c Restrict dot_sparse to sm_80-sm_89
a2558ad19 Add lit tests for dot_sparse and fix min sparse dot size
c070f10f8 Verify the dot_sparse metadata operand
1aeeed9af Use LinearEncoding for dot_sparse metadata instead of a new attr
302982abe Add tt.dot_sparse and MMAv2 lowering for 2:4 structured sparsity
```

**목차**

1. [전체 흐름](#1-전체-흐름)
2. [메타데이터 포맷](#2-메타데이터-포맷)
3. [LinearEncodingAttr](#3-linearencodingattr)
4. [getSparseMetadataLayout](#4-getsparsemetadatalayout)
5. [프론트엔드와 검증](#5-프론트엔드와-검증)
6. [SparseBlockedToMMA](#6-sparseblockedtomma)
7. [PTX 생성](#7-ptx-생성)
8. [인터프리터와 백엔드 설정](#8-인터프리터와-백엔드-설정)
9. [MMAv3 — wgmma.mma_async.sp](#9-mmav3--wgmmamma_asyncsp)
10. [MMAv5 — tcgen05.mma.sp](#10-mmav5--tcgen05mmasp)

---

## 1. 전체 흐름

커널 소스의 `tl.dot_sparse(a, b, meta, acc)` 한 줄이 PTX 명령 하나가 되기까지 여섯 단계를 지난다.
각 단계가 담당하는 것은 서로 겹치지 않는다.

| 단계 | 담당 | 위치 |
|---|---|---|
| 빌트인 | 인자를 semantic 으로 넘김 | `language/core.py` |
| Semantic | shape · dtype 검증, 결과 타입 결정, op 생성 | `language/semantic.py` |
| IR (TTIR) | `tt.dot_sparse` + verifier | `TritonOps.td`, `Triton/IR/Ops.cpp` |
| 레이아웃 (TTGIR) | MMA 인코딩 선택, 피연산자·메타데이터 레이아웃 부착 | `AccelerateMatmul.cpp` |
| Lowering | 레지스터/TMEM 배치 → sparse MMA PTX | `DotOpToLLVM/{MMAv2,WGMMA,MMAv5}.cpp` |
| 인터프리터 | GPU 없이 같은 의미로 실행 | `runtime/interpreter.py` |

설계의 핵심 원칙 하나: **메타데이터의 사용자 계약은 아키텍처와 무관하게 고정**이고,
아키텍처별 차이(v2 는 레지스터, v3 는 warp basis 재정렬, v5 는 TMEM 상주)는 전부 컴파일러 내부로
숨긴다. 사용자는 어떤 칩에서든 같은 `meta` 텐서를 만들고, `12-dot-sparse.py` 튜토리얼이
sm_86 / sm_90 / sm_100 / sm_120 에서 한 글자도 다르지 않게 돈다.

`DotSparseOp` 는 아키텍처 선택 지점에서 세 갈래로 갈린다:

```
tt.dot_sparse
 ├─ SparseBlockedToMMA    sm_80–89, sm_120+ → tt.dot_sparse(#mma v2)  → mma.sp.sync
 │                        sm_90 (형상 미지원 시 폴백)
 ├─ SparseBlockedToMMA    sm_90            → ttng.warp_group_dot + meta → wgmma.mma_async.sp
 └─ SparseBlockedToMMAv5  sm_100–119 (M≥128) → ttng.tc_gen5_mma + meta   → tcgen05.mma.sp
```

---

## 2. 메타데이터 포맷

2:4 희소성은 dense 원소 4개마다 2개만 남긴다. `a` 는 남은 것만 담으므로
마지막 차원이 dense K 의 절반이고, **어느 2개를 남겼는지**를 `meta` 가 기술한다.

| 텐서 | shape | dtype |
|---|---|---|
| `a` (sparse lhs) | `[…, M, K_dense/2]` | f16 / bf16 / i8 / e4m3 / e5m2 |
| `b` (dense rhs) | `[…, K_dense, N]` | 같음 (fp8 은 혼합 허용) |
| `meta` | `[…, M, K_dense/16]` | **i16** |

`i16` 하나가 4비트 그룹 4개를 담고, 각 그룹이 dense 4원소 그룹 하나를 기술한다.
따라서 `i16` 하나가 dense 원소 16개, 즉 `a` 의 원소 8개를 덮는다
→ `meta` 의 마지막 차원은 `a` 의 마지막 차원 ÷ 8.

```
i16 한 개 (16비트)
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  nibble 3    │  nibble 2    │  nibble 1    │  nibble 0    │
│  bits 15:12  │  bits 11:8   │  bits 7:4    │  bits 3:0    │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ dense 12–15  │ dense 8–11   │ dense 4–7    │ dense 0–3    │
└──────────────┴──────────────┴──────────────┴──────────────┘

nibble 한 개 (4비트)
                             ┌──────────────┬──────────────┐
                             │  bits 3:2    │  bits 1:0    │
                             │ 2번째 남긴   │ 1번째 남긴   │
                             │ 원소 idx     │ 원소 idx     │
                             └──────────────┴──────────────┘
```

### 구체적 예

dense 4원소 중 0번과 3번을 남겼다면 nibble = `0b1100` = `0xC`
(bits 1:0 = 0, bits 3:2 = 3). 이 nibble 이 그룹 0 이면 `i16` 의 최하위 4비트에 놓인다.

```python
# 튜토리얼 compress() 가 하는 일 — 포맷의 정의 그 자체
for o in range(0, len(flat), 4):      # dense 4원소 그룹마다
    v = c = 0
    for i in range(4):
        if flat[o+i] != 0:
            v |= i << (2 * c); c += 1     # 2비트 인덱스를 c번째 자리에
    nib.append(v)                          # nibble 완성
for o in range(0, len(nib), 4):       # nibble 4개 = i16 하나
    metas.append(sum(nib[o+i] << (4*i) for i in range(4)))
```

> **이 포맷은 dtype 과 무관하다.** 메타데이터는 dense 원소당 2비트이므로
> fp8 이나 int8 (`m16n8k64`) 로 확장해도 `[…, M, K_dense/16]` i16 계약이 그대로 유지된다.
> 달라지는 것은 스레드 간 분배뿐이고, 그것은 컴파일러 내부다.

---

## 3. LinearEncodingAttr

Triton 의 모든 분산 레이아웃은 **LinearLayout** 으로 수렴한다.
LinearLayout 은 입력 차원 `{register, lane, warp, block}` 의 비트를
출력 차원 `{dim0, dim1, …}` 의 좌표로 보내는 GF(2) 선형 사상이고,
각 입력 비트가 어디로 가는지를 **basis 벡터** 하나가 지정한다.

`LinearEncodingAttr` 는 그 LinearLayout 을 감싸는 attribute 다. 요구사항은 두 가지뿐이고,
이 문서 전체에서 가장 중요한 사실이 그 안에 있다.

```cpp
// Dialect.cpp — LinearEncodingAttr::verify
if (failed(verifyDistributedLinearLayoutDims(emitError, linearLayout)))
  return failure();
if (!isPermutationMatrixLayout(linearLayout))
  return emitError()
         << "LinearEncodingAttr requires a permutation matrix layout "
            "after removing broadcast bases";

// Dialect.cpp:83
bool isPermutationMatrixLayout(const LinearLayout &ll) {
  if (!hasPowerOfTwoBases(ll)) return false;   // basis 마다 비영 성분 ≤ 1
  LinearLayout flattened = ll.flattenIns().flattenOuts();
  auto inDim = *flattened.getInDimNames().begin();
  return flattened.removeZeroBasesAlongDim(inDim).isInvertible();
}                                              // ↑ zero basis 제거 후 전단사
```

두 조건을 풀어 쓰면:

1. **모든 basis 는 비영 성분이 최대 하나** — 즉 하나의 출력 차원만 따라 움직이거나,
   아예 전부 0 (**broadcast basis**) 이다.
2. **broadcast basis 를 제거한 뒤** 전단사여야 한다.

> **zero basis 가 명시적으로 허용된다**는 것이 결정적이다. 어떤 입력 비트의 basis 가 전부 0 이면
> 그 비트를 바꿔도 좌표가 같은 곳을 가리킨다 — 즉 **여러 스레드·레지스터가 같은 데이터를 본다(복제)**.
> 전단사 요구는 그 비트들을 *빼고 난 뒤에만* 적용되므로, 복제가 있는 레이아웃도 이 attribute 로 표현된다.

처음 구현에서는 전용 `SparseMetadataEncodingAttr` 를 만들고 그 설명에
*"cannot be expressed by a bijective LinearEncoding"* 이라고 적었다.
메타데이터가 N-warp 방향으로 복제되기 때문이라는 이유였는데, 위 두 조건을 보면
그 복제가 정확히 broadcast basis 로 표현된다. 전제가 틀렸으므로 attribute 를 제거했다 (`1aeeed9af`).

참고로 upstream 에는 더 느슨한 `GenericLinearEncodingAttr` 도 있다.
warp basis 의 swizzle 을 허용하고 전단사 대신 surjective 만 요구한다. 메타데이터는 그것까지 필요하지 않았다.

복제를 실제로 풀어주는 런타임 쪽 장치도 이미 존재한다.
`unpackTensorElements` 는 LLVM struct 에 담긴 *고유* 원소만 꺼낸 뒤
레이아웃에 따라 확장한다 — 그래서 lowering 이 복제를 따로 처리할 필요가 없다.

```cpp
// Conversion/TritonGPUToLLVM/Utility.cpp:1054
SmallVector<Value> unpackTensorElements(loc, llvmStruct, rewriter, originalType) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(originalType))
    return broadcastAs(unpackUniqueTensorElements(loc, llvmStruct, rewriter),
                       triton::gpu::toLinearLayout(tensorTy));
  return unpackLLElements(loc, llvmStruct, rewriter);
}
```

---

## 4. getSparseMetadataLayout

**위치**: `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`

```cpp
LinearLayout getSparseMetadataLayout(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                     ArrayRef<unsigned> warpsPerCTA,
                                     CGAEncodingAttr cgaLayout,
                                     unsigned elemBitWidth,
                                     bool rowMajorWarpOrder = true);
```

이 경로의 심장부다. `mma.sp` 하드웨어가 요구하는 스레드↔메타데이터 대응을 basis 벡터로 기술한다.
MMAv3(`wgmma.mma_async.sp`)도 **같은 함수를 쓴다** — 아래 register/lane basis 는 그대로이고
`rowMajorWarpOrder` 만 `false` 가 된다 ([§9](#9-mmav3--wgmmamma_asyncsp)).
`elemBitWidth` 가 명령을 고른다 — 16비트는 `m16n8k32`, 8비트는 `m16n8k64`.
**두 명령의 매핑은 비트 하나 차이가 아니라 구조가 다르다.**

비트 회계가 그 이유를 설명한다. i16 하나가 dense 16개를 덮으므로 MMA 하나가 쓰는 metadata 열은
`K_dense/16` 열이고, 필요한 총 비트는 `16행 × 열수 × 16비트`다. warp 이 공급할 수 있는 양은
`32 lane × 32비트 = 1024비트`다.

| 명령 | 열/ MMA | 필요 비트 | warp 용량 대비 |
|---|---|---|---|
| `m16n8k32` (16비트) | 2 | 512 | **절반** → lane 비트 하나가 남음 |
| `m16n8k64` (8비트) | 4 | 1024 | **꽉 참** → 남는 lane 비트 없음 |

```
m16n8k32:  row = T>>2,             col = T%2       (T%4 상위 비트는 broadcast)
           E[15:0] = meta[row][col],  E[31:16] = meta[row + 8][col]

m16n8k64:  row = (T>>2) + 8*(T&1),  col = 2*((T>>1)&1)
           E[15:0] = meta[row][col],  E[31:16] = meta[row][col + 1]
```

즉 16비트는 한 스레드가 **같은 열의 두 행**(r, r+8)을 담고, 8비트는 **같은 행의 인접한 두 열**을
담는다. `+8` 행 오프셋이 레지스터에서 lane 비트 0 으로 옮겨가고, 레지스터 비트는 열 +1 이 된다.

> **이 매핑은 추측이 아니라 실측이다.** 처음엔 "남은 lane 비트가 열 비트가 된다"(= 16비트 매핑의
> 자연스러운 확장)로 구현했고 int8 수치가 전부 틀렸다. 비트 회계로 가능한 lane 비트 배정이
> 여러 개라서, 후보를 하나씩 빌드하는 대신 **하드웨어에 직접 물어보는 프로브**를 썼다:
> 각 i16 을 `0x4444`(그룹마다 0,1 유지) 또는 `0xEEEE`(2,3 유지) 로 채워 텐서 원소마다 1비트를
> 심고, `B` 를 dense 행 하나만 1인 one-hot 으로 두면 `D[:, 0]` 이 하드웨어가 복원한 dense 열을
> 그대로 보여준다. 텐서 원소 인덱스를 6비트로 이진 인코딩해 6라운드 돌리면 64개 슬롯의
> 대응이 한 번에 나온다. 결과가 위 식이고, 그 뒤 int8/fp8 이 비트 단위로 일치했다.

### basis 구성 규칙

| 입력 차원 | m16n8k32 | m16n8k64 | 의미 |
|---|---|---|---|
| register 0 | `[8, 0]` | `[0, 1]` | 스레드의 두 번째 원소 (E 상위 16비트) |
| register | `[0, 2·2^i]` | `[0, 4·2^i]` | K 그룹 확장 — MMA 하나가 쓰는 열 수를 넘는 부분 |
| register | `[16·w_M·2^i, 0]` | 같음 | M 방향 반복 — warp 커버리지를 넘는 행 |
| lane 0 | `[0, 1]` (col) | `[8, 0]` (row+8) | |
| lane 1 | **`[0, 0]`** broadcast | `[0, 2]` (col 상위) | 8비트는 남는 lane 비트가 없다 |
| lane 2–4 | `[1,0] [2,0] [4,0]` | 같음 | 행 = T>>2 |
| warp | **`[0, 0]` …** | 같음 | **broadcast** — N-warp 는 같은 메타데이터를 씀 |
| warp | `[16·2^i, 0]` | 같음 | M-warp 는 16행씩 나눠 담당 |

> warp basis 의 **순서가 N 먼저, M 나중**인 것은 임의가 아니다.
> `NvidiaMmaEncodingAttr` 가 `warpOrder = getMatrixOrder(rank=2, rowMajor=true) = {1, 0}` 을
> 쓰기 때문에 warp 비트 0 이 dim1(N), 비트 1 이 dim0(M) 을 가리킨다.
> 이 순서를 뒤집으면 조용히 틀린 값이 나온다.

### 실제 출력 — meta `[128, 4]`, warpsPerCTA `[2, 2]`

`triton-opt --tritongpu-accelerate-matmul` 이 뱉는 인코딩. 위 규칙을 대입한 결과와 정확히 일치한다.

```mlir
#linear = #ttg.linear<{
  register = [[8, 0], [0, 2], [32, 0], [64, 0]],
  lane     = [[0, 1], [0, 0], [1, 0], [2, 0], [4, 0]],
  warp     = [[0, 0], [16, 0]],
  block    = []}>
```
(`lane` 의 두 번째와 `warp` 의 첫 번째가 broadcast basis)

비영 basis 는 `4 + 4 + 1 = 9` 개이고 `2^9 = 512 = 128 × 4` —
broadcast basis 두 개를 빼면 정확히 전단사다. `LinearEncodingAttr` 의 두 조건이 모두 만족된다.

> 리팩터 전(전용 attr)과 후(`LinearEncodingAttr`)의 수치 결과가
> **7개 형상 전부에서 비트 단위로 동일**했다.
> "다른데 우연히 맞는 레이아웃"이 아니라 같은 레이아웃이라는 확인이다.

---

## 5. 프론트엔드와 검증

검증은 두 계층이다. 사용자 오류는 semantic 계층에서 소스 위치가 붙은
`CompilationError` 로 잡고, IR 레벨 불변식은 verifier 가 백스톱으로 지킨다.
upstream 의 `dot_scaled` 가 scale 을 검증하는 구조와 같다.

### `TritonSemantic.dot_sparse`

**위치**: `python/triton/language/semantic.py`

검사 순서와 각 검사가 잡는 것:

| 검사 | 실패 시 메시지 |
|---|---|
| 타겟이 sparse dot 을 지원하는가 | `Sparse dot is unsupported on this platform` |
| `out_dtype` 가 f32/f16(정수는 i32) 인가 | `out_dtype=… is unsupported for dot_sparse` |
| f16 누산기를 골랐다면 입력이 fp16 인가 | `out_dtype=float16 requires float16 inputs` |
| dtype 이 지원되는가 (아키텍처 미지원도 여기서 걸림) | `Unsupported lhs dtype … for dot_sparse on this target` |
| meta rank == lhs rank | `Metadata must have the same rank as the first input` |
| meta dtype == i16 | `Metadata must be int16` |
| meta shape == `[…, M, K_a/8]` | `Metadata must be a tensor of shape …` |
| `K_a · 2 == K_b` | `… are not compatible for matmul` |
| M/N/K 하한 (`min_sparse_dot_size`) | `Input shapes should have M ≥ …, N ≥ …, K ≥ …` |

결과 타입은 `[…, M, N]` 이고 스칼라 타입은 i8 입력이면 i32, 그 밖에는 `out_dtype`
(기본 fp32, fp16 입력에 한해 fp16 선택 가능)이다. `acc` 가 `None` 이면 0 으로 splat 한다.
마지막 반환은 `tl.tensor` 가 아니라 `self.tensor` 를 쓴다 —
`TritonSemantic.tensor` 는 `GluonSemantic` 이 오버라이드하는 클래스 속성이므로,
직접 참조하면 Gluon 경로에서 잘못된 타입이 만들어진다.

### `DotSparseOp::verifyDims`

**위치**: `lib/Dialect/Triton/IR/Ops.cpp`

`DotOpInterface` 의 기본 구현은 `K_a == K_b` 를 본다.
sparse 는 `a` 가 절반만 담으므로 이 메서드를 오버라이드한다.
TableGen 쪽에 `DeclareOpInterfaceMethods<DotOpInterface, ["verifyDims"]>` 로
오버라이드를 명시해야 한다 — 목록을 빼면 기본 구현이 쓰이고, 이 정의는 선언이 없어 빌드가 깨진다.

```cpp
bool DotSparseOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  return aShape[aShape.size() - 1] * 2 == bShape[bShape.size() - 2];
}
```

### `DotSparseOp::verify`

**위치**: `lib/Dialect/Triton/IR/Ops.cpp`

`DotOpInterface` 의 verifier 는 `$a` · `$b` · `$c` 의 rank, batch, 출력 shape 를 이미 검사한다.
하지만 `$aMeta`(operand 3)는 **전혀 보지 않는다** — 그래서 이 메서드가 필요하다.
메타 element type(i16), rank 일치, batch 일치, M 일치, 그리고 `metaK == K_a/8` 을 확인한다.

---

## 6. SparseBlockedToMMA

**위치**: `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`

blocked 레이아웃의 `tt.dot_sparse` 를 MMAv2 또는 MMAv3 레이아웃으로 바꾸는 rewrite 패턴
(MMAv5 는 별도 패턴 `SparseBlockedToMMAv5`, [§10](#10-mmav5--tcgen05mmasp)). 네 가지 일을 한다.

1. **버전 선택** — `getSparseMMAVersion(computeCapability, dotOp)`:
   `[80,90)` 과 `[120,130)` 은 2, `[90,100)` 은 형상이 맞으면 3 아니면 remark + 2,
   그 밖(= `[100,120)`)은 0 = 변환하지 않음 (v5 패턴이 받는다).
   **먼저 `isa<NvidiaMmaEncodingAttr>` 로 이미 변환된 op 을 걸러낸다** — 순서를 바꾸면
   자기가 만든 op 에 remark 를 한 번 더 남긴다.
2. **MMA 인코딩 생성** — `createMMAEncodingForDot(…, versionMajor, /*isSparse=*/true)`.
   v3 에서 instrShape 의 K 가 dense K 로 두 배가 되는 지점이다.
3. **A/B 피연산자 변환** — v2 는 `convertDotOperandForMMA`(dot operand 인코딩),
   v3 는 `getSharedMemoryMMAOperand`(SMEM).
4. **메타데이터 레이아웃 부착** — 아래 코드. 두 버전이 공유한다.

```cpp
auto ll = triton::gpu::getSparseMetadataLayout(
    ctx, metaTy.getShape(), mmaResult.mmaEnc.getWarpsPerCTA(),
    mmaResult.mmaEnc.getCGALayout(), static_cast<unsigned>(minBitwidth),
    /*rowMajorWarpOrder=*/!mmaResult.mmaEnc.isHopper());
auto metaEncoding = triton::gpu::LinearEncodingAttr::get(ctx, std::move(ll));
auto newMetaTy = metaTy.cloneWithEncoding(metaEncoding);
aMeta = ConvertLayoutOp::create(rewriter, aMeta.getLoc(), newMetaTy, aMeta);
```

`getSM120DotScaledScaleLayout` 이 scaled dot 의 scale 을 다루는 방식과 같은 형태다 —
자유 함수가 LinearLayout 을 만들고, `LinearEncodingAttr::get` 이 감싸고,
`ConvertLayoutOp` 가 실제 재배치를 맡는다. 재배치 비용은 A/B 피연산자가
이미 치르는 것과 같은 종류이므로 새로운 비용이 아니다.

> **bitwidth 를 직접 읽는 이유.** 일반 경로의 `computeOrigBitWidth` 는
> backward slice 에 `JoinOp` 가 있으면 bitwidth 를 절반으로 본다(fp8 쌍을 fp16 으로 로드하는 경우 보정).
> 그런데 `dot_sparse` 사용자는 sparse 값을 끼워넣으려고 `tl.join` 을 쓸 수 있고
> (fused sparsify_24 + dot_sparse 같은 커널), 그때 이 절반 처리는 틀린 kWidth 를 만든다.
> 그래서 element type 의 bitwidth 를 그대로 쓴다.

---

## 7. PTX 생성

### `getMmaTypeSparseDot`

**위치**: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`

`(aTy, bTy, dTy)` 조합을 `TensorCoreType` 으로 매핑한다. f32 누산기에 f16/bf16/fp8,
i32 누산기에 i8 을 인정하고 그 외는 `NOT_APPLICABLE` 이다.
PTX 문자열 테이블은 이 열거형으로 인덱싱된다.

```cpp
static const std::map<TensorCoreType, std::string> mmaInstrPtxSparse = {
  {FP32_FP16_FP16_FP32, "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"},
  {FP32_BF16_BF16_FP32, "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"},
  {INT32_INT8_INT8_INT32,
   "mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32"},
  {FP32_FP8E4M3FN_FP8E4M3FN_FP32,
   "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32"},  // + e4m3/e5m2 혼합 4종
};
```

8비트에서 달라지는 것은 이 표와 metadata 레이아웃, 그리고 두 가지뿐이다.

- **누산기 제약**: i32 와 packed f16x2 누산기는 `=f` 가 아니라 `=r`
  (`callMmaSparse` 의 `constraintRet`). dense 경로의 `isIntMMA || isAccF16 ? "=r" : "=f"` 와 같다.
- **프론트엔드 결과 타입**: `semantic.dot_sparse` 가 i8 입력에 `tl.int32` 를 준다 (dense `dot` 과 동일).

fp16 입력에는 f16 누산 변종도 있다:

```cpp
  {FP16_FP16_FP16_FP16, "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16"},
```

`out_dtype=tl.float16` 일 때만 선택되며 (기본은 f32), fp32 누산 MMA 가 절반 속도인 칩
(지금까지의 모든 컨슈머 파트)에서 명령 처리율이 2배다. PTX 에 **fp16 입력 전용**이라 bf16·fp8 은
막는다 — ptxas 원문: *"Instruction 'mma with with F16 accumulator and F8 floating point type'
not supported"*. 정밀도를 절반 내주는 옵션이므로 절대 암묵적으로 고르지 않는다.

A/B 레지스터 개수(각 4개)와 `repKB == 2·repKA`, B 의 k-slot 순서는 16비트와 **같다** —
8비트는 레지스터당 원소가 4개로 늘어나면서 명령의 K 도 2배가 되므로 회계가 그대로 성립한다.
`unsigned` 여부는 IR 에 없다(Triton 정수는 signless) → dense int8 dot 처럼 `.s8` 만 쓴다.

### `convertMMASparseDot`

Lowering 의 본체. dense 경로와 다른 지점만 짚으면:

- **A 와 B 의 K 반복 횟수가 다르다.** `repKA` 는 압축된 K, `repKB` 는 dense K 기준이라
  `repKB == 2 · repKA`. MMA 한 번이 A 타일 1개와 B 타일 2개를 소비한다.
- **메타데이터를 i32 로 조립한다.** 레이아웃이 준 i16 두 개를 합쳐
  하위 16비트에 `row`, 상위 16비트에 `row + 8` 을 놓는다.

```cpp
int m_k = m * repKA + k;
int idx0 = m_k * 2;      // 하위 16비트 (row)
int idx1 = m_k * 2 + 1;  // 상위 16비트 (row + 8)
Value meta0 = tb.zext(i32Ty, metaElems[idx0]);
Value meta1 = tb.zext(i32Ty, metaElems[idx1]);
metaRegs[m][k] = tb.or_(meta0, tb.shl(meta1, tb.i32_val(16)));
```

원소를 꺼낼 때 `unpackLLElements` 가 아니라
`unpackTensorElements(…, op.getAMeta().getType())` 를 쓴다.
후자가 레이아웃을 보고 broadcast 를 확장하므로, [§3](#3-linearencodingattr) 의 복제가 여기서 자동으로 처리된다.

### `callMmaSparse`

`mma.sp` 한 개를 방출한다. 피연산자 구성:

| 피연산자 | 개수 | 구성 |
|---|---|---|
| D (결과) | 4 × f32 | 누산기와 같은 자리 |
| A | 4 × b32 | k-reg 2개 × m-tile 2 위치 |
| B | 4 × b32 | B 의 k-group 2개 × k-reg 2개 (K=32 커버) |
| C (누산기) | 4 × f32 | `fc` 에서 인덱싱 |
| 메타데이터 | 1 × b32 | 위에서 조립한 `metaRegs[m][k]` |
| selector | 상수 | `0x0` |

> **selector 가 항상 0 인 근거.** selector 는 quad 의 어느 스레드가 quad 전체의 메타데이터를
> 공급할지 고른다. `getSparseMetadataLayout` 이 모든 스레드에게 자기 행의 메타데이터를 주므로
> quad 의 0번 스레드가 이미 필요한 값을 들고 있다 — 따라서 0 이다.
> 8비트(`m16n8k64`)는 warp 의 32개 lane 이 전부 서로 다른 메타데이터를 공급해야 하므로
> 애초에 고를 여지가 없다(0 만 유효). 레이아웃과 이 상수는 한 쌍이므로 한쪽만 바꾸면 안 된다.

니모닉은 **아키텍처에 따라 갈린다** (`sparseMmaOpcode()`):

```cpp
if (computeCapability < 100)
  return instr;                                        // mma.sp
return "mma.sp::ordered_metadata" + instr.substr(6);   // Blackwell
```

`ordered_metadata` 는 같은 명령에 "각 4비트 그룹의 두 2비트 인덱스가 정렬돼 있다" 는 약속을
붙인 것이고, 그래서 허용 metadata 가 `{0100,1000,1100,1001,1101,0110,1110}` 로 제한된다.
튜토리얼 `compress()` 가 인덱스를 증가 순서로 채우므로 생성 가능한 6개 값이 전부 그 안이라
**인코딩은 바뀌지 않는다** — 수치가 bit-exact 로 유지되는 것으로 확인했다.

sm_80–89 는 legacy 를 유지한다. 거기서는 측정 이득이 0 이고 PTX 하한만 8.5 로 올라간다.
Blackwell 에서는 얘기가 다르다 — ptxas advisory 가 문자 그대로였다:

| RTX 5080, 2048³ | `mma.sp` | `sp::ordered_metadata` |
|---|---|---|
| fp16 | 35.8 | **143.1** (4.00x) |
| int8 | 58.3 | **357.2** (6.13x) |
| bf16 / fp8 | 143.5 / 261.0 | 변화 없음 |

fp16·int8 만 벌칙을 받고 bf16·fp8 은 이미 빠른 경로였다. 고치기 전엔 sparse fp16/int8 이
같은 칩의 dense 보다 느렸다. MMAv3 는 해당 없다 — ptxas 가
`wgmma.mma_async.sp::ordered_metadata` 를 illegal modifier 로 거부하고 advisory 도 안 낸다.

---

## 8. 인터프리터와 백엔드 설정

### `InterpreterBuilder.create_dot_sparse`

**위치**: `python/triton/runtime/interpreter.py`

`TRITON_INTERPRET=1` 경로. 메타데이터로 dense lhs 를 복원한 뒤 dense matmul 을 재사용한다.
GPU lowering 과 동일 입력에서 `1.9e-6` 까지 일치하며,
메타데이터 포맷의 **실행 가능한 명세** 역할도 한다.

```python
k_sparse = a_data.shape[-1]
# 각 남긴 원소가 속한 4원소 그룹, 그리고 그 그룹을 기술하는 i16 과 nibble
group  = np.arange(k_sparse) // 2
nibble = (meta[..., group // 4] >> (4 * (group % 4)).astype(np.uint16)) & 0xF
# nibble 안의 두 2비트 인덱스
idx_in_group = (nibble >> (2 * (np.arange(k_sparse) % 2)).astype(np.uint16)) & 0x3
dense_col = (group * 4 + idx_in_group).astype(np.intp)
dense = np.zeros(dense_shape, dtype=a_data.dtype)
np.put_along_axis(dense, dense_col, a_data, axis=-1)
```

`group // 4` 와 `group % 4` 의 4 는 "i16 하나가 그룹 4개"라는 뜻이다.
여기를 2 로 잘못 쓰면 인덱스가 범위를 벗어나거나 조용히 엉뚱한 그룹을 읽는다.

### `get_min_sparse_dot_size`

**위치**: `third_party/nvidia/backend/compiler.py`

허용되는 최소 `(M, N, K)`. 반환하는 K 는 **dense K**, 즉 sparse lhs 의 K 의 두 배다.
`m16n8k32` 는 `(16, 8, 32)`, 8비트 `m16n8k64` 는 `(16, 8, 64)`.
원래 16비트에서 `(16, 16, 16)` 을 반환했는데, K 하한이 너무 약해 사실상 아무것도 거르지 못했다.

**sm_100–119 는 M 하한이 128 이다.** MMAv5 는 M=64 에서 metadata TMEM 레이아웃이 달라
(`SparseBlockedToMMAv5` 가 매치하지 않음) 폴백도 없어서, 그냥 두면 프론트엔드를 통과한 뒤
`PassManager::run failed` 로 죽는다. 여기서 막아 *"Input shapes should have M >= 128"* 를 준다.
sm_90 은 폴백(MMAv2)이 있으므로 하한을 올리지 않는다.

### `get_supported_sparse_dot_dtypes`

**위치**: `third_party/nvidia/backend/compiler.py`

`80 ≤ capability < 130` 에서 dtype 을 인정한다. `int8` 은 전 범위,
`fp8e4nv` / `fp8e5` 는 **sm_89+** (ptxas 가 그 아래에서 `m16n8k64` 의 fp8 변종을 거부한다),
`fp16` / `bf16` 은 `100 ≤ cc < 120` 만 제외한다 (v5 16비트 metadata TMEM packing 미표현,
[§10](#10-mmav5--tcgen05mmasp)).
그래서 이 함수는 dtype 게이트이면서 동시에 아키텍처 게이트다 —
semantic 의 에러 메시지가 "on this target" 이라고 말하는 이유다.

---

## 9. MMAv3 — `wgmma.mma_async.sp`

### metadata 매핑은 v2 와 같다 (추정이 아니라 문서 사실)

PTX ISA 의 §9.7.16.6.2 *"Matrix fragments for warpgroup-level multiply-accumulate operation with
sparse matrix A"* 는 metadata 레이아웃 그림으로 **`mma.sp` 절과 같은 이미지 파일**을 가리킨다:

| wgmma.sp 그림 | 이미지 파일 | 같은 파일을 쓰는 mma.sp 그림 |
|---|---|---|
| Figure 176 (`m64nNk32`, f16/bf16) | `sparse-mma-metadata-16832-f16bf16.png` | `m16n8k32` metadata |
| Figure 180 (`m64nNk64`, 열 0–31) | `sparse-mma-metadata-16864-u8s8-first32col.png` | `m16n8k64` 열 0–31 |
| Figure 181 (`m64nNk64`, 열 32–63) | `sparse-mma-metadata-16864-u8s8-last32col.png` | `m16n8k64` 열 32–63 |

즉 **warp 내부 thread↔metadata 대응이 정의상 동일**하다. 그림을 직접 판독해 §4 의 식과
일치함도 재확인했다 (16비트: 행 0 이 `T_2i` bits[15:0], 행 8 이 `T_2i` bits[31:16], 행 1 이 `T_2i+4`
→ `row = T>>2`, `col = T%2`, selector `i` 가 quad 내 thread pair 선택. 8비트: 행 0 이 `T_0` bits[31:0]
= dense 0–31, 행 8 이 `T_1` → `row = (T>>2) + 8*(T&1)`, `col = 2*((T>>1)&1)`).

`m64nNk64` 의 selector 는 0 만 유효하고 `m64nNk32` 는 0/1 이다 (ptxas 로 전수 확인).
비트 회계도 그것을 말한다 — warpgroup 용량 `128 × 32 = 4096`비트에 대해
16비트는 `64행 × 32 dense × 1비트 = 2048`(절반 → selector 1비트), 8비트는 `64 × 64 = 4096`(꽉 참).

### 유일한 차이 — warp basis 순서

warpgroup 은 4 warp 을 **M 방향**으로 쌓는다 (ISA: *"warp `%warpid % 4 = k` supplies sparsity
information for rows `16k .. 16k+15`"*). `NvidiaMmaEncodingAttr::toLinearLayout` 이 Hopper 에서
`warpOrder = getMatrixOrder(rank, rowMajor = !isHopper()) = {0, 1}` 를 쓰는 것과 정확히 맞물린다.

```cpp
// getSparseMetadataLayout 에 파라미터 하나가 늘었다
LinearLayout getSparseMetadataLayout(..., unsigned elemBitWidth,
                                     bool rowMajorWarpOrder = true);
// v2: N-warp broadcast 먼저, 그 다음 M-warp (16, 32, ...)
// v3: M-warp 먼저, 그 다음 N-warp broadcast
```

`SparseBlockedToMMA` 는 `!mmaResult.mmaEnc.isHopper()` 를 그대로 넘긴다. 뒤집으면 조용히 틀린다.

실제 출력 (`meta[128,4]`, warpsPerCTA `[4,1]`):

```mlir
#linear = #ttg.linear<{
  register = [[8, 0], [0, 2], [64, 0]],
  lane     = [[0, 1], [0, 0], [1, 0], [2, 0], [4, 0]],
  warp     = [[16, 0], [32, 0]],   // ← v2 라면 [[0,0]...] 이 먼저 온다
  block    = []}>
```

### instrShape 의 K 는 dense K

v3 는 v2 와 달리 `instrShape` 에 K 가 있고, PTX 니모닉의 `k32`/`k64` 가 그것이다.
sparse 는 같은 A 바이트로 dense K 의 두 배를 먹으므로 K 가 두 배다.

```cpp
// Transforms/Utility.cpp
unsigned k = (isSparse ? 512 : 256) / eltType.getIntOrFloatBitWidth();
```

lowering(`WGMMA.cpp`)에서 절반이 되는 것은 **A 타일과 A descriptor 뿐**이다:

| 값 | dense | sparse |
|---|---|---|
| `mmaSizeK` (= PTX 의 `k`) | 16 / 32 | 32 / 64 |
| `mmaSizeKA` (A 타일 폭, descriptor 스텝) | 같음 | **절반** |
| B 타일 높이 / B descriptor 스텝 | `mmaSizeK` | `mmaSizeK` (그대로) |
| `numRepK` | `aShape[1] / mmaSizeK` | `aShape[1] / mmaSizeKA` |
| `numLowPrecisionAcc += K` | dense K | dense K (그대로) |

즉 sparse `m64nNk32` 의 A SMEM 타일은 dense `m64nNk16` 의 것과 **완전히 같다**. 좋은 정합성 신호다.

### 새 op 이 아니라 optional 피연산자

`ttng.warp_group_dot` 에 `Optional<TT_IntTensor>:$aMeta` 를 붙였다. 새 op 을 만들면
`WGMMAPipeline.cpp`(23곳) · `WSDataPartition.cpp`(10곳) · `OptimizeAccumulatorInit.cpp` 등
15개 파일 60여 곳을 손대야 하는데, optional 피연산자는 그 전부를 그대로 통과한다.

```mlir
%acc = ttng.warp_group_dot %a meta %meta, %b, %acc0 {maxNumImpreciseAcc = 1073741824 : i32}
  : !ttg.memdesc<128x32xf16, #shared, #smem> meta tensor<128x4xi16, #linear>
  * !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<128x128xf32, #mma>
```

> `AttrSizedOperandSegments` 를 추가해도 **출력 IR 은 바뀌지 않는다** — ODS 가 생성하는 프린터가
> `operandSegmentSizes` 를 elide 한다. 기존 lit 테스트가 전부 그대로 통과하는 이유다.

주의한 것 세 가지:

- **sparse lhs 는 SMEM 고정.** 레지스터 lhs 변종(`wgmma.mma_async.sp ... d, a, b-desc, ...`)이
  PTX 에 있지만, 그걸 쓰는 이유인 in-register 파이프라이닝 `splitRSDot` 이 K 를 반으로 쪼개므로
  metadata 도 같이 쪼개야 한다. `splitRSDot` 에 `dotOp.isSparse()` 가드를 넣고 SMEM 으로 고정했다.
- **sp-meta 는 fence 앞에서 만든다.** PTX: *"wgmma.fence must be used to fence the register
  accesses of wgmma.mma_async from their prior accesses."* sp-meta 도 레지스터 접근이므로
  레지스터 A 와 같은 취급이 필요하다. `buildSparseMetaRegs` 를 `WgmmaFenceAlignedOp` 생성 **전**에
  호출한다.
- **`max_num_imprecise_acc`.** `tt.dot_sparse` 에는 그 attribute 가 없다(#6714 와 공유하는 op).
  32 미만이면 fp8 입력 + f32 누산이 `WarpGroupDotOp::verify()` 에서 거부되므로,
  패스가 sm_90 dense 기본값 `1 << 30`(= 네이티브 누산)을 지정한다.

생성된 PTX (fp16, BLOCK 128×128×64, 4 warp → wgmma.sp 4개):

```
wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.f16.f16
    {%r434,...,%r497},   // D: N/2 = 64개 f32
    %rd173, %rd174,      // a-desc, b-desc
    %r216, 0,            // sp-meta, sp-sel
    %p2, 1, 1, 0, 1;     // scale-d, imm-scale-a/b, trans-a/b
```

> `%r216..%r219` 는 `ldmatrix.trans.b16` 결과 레지스터 **그대로**다. lowering 이 i16 두 개를
> `zext/shl/or` 로 조립하는데, 그 둘이 같은 32비트 워드의 하위/상위 절반이므로 LLVM 이 왕복을
> 접어버린다. 레이아웃이 의도대로 붙었다는 강한 증거다.

### 기본 비활성 상태로 출하된다

실기 수치 검증 전이라 `knobs.nvidia.enable_unverified_sparse_wgmma`
(`TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1`) 로만 열린다.
`get_supported_sparse_dot_dtypes` 가 `[90,100)` 에서 knob 을 보고 거부하므로
Hopper 사용자는 평범한 "unsupported on this target" 을 받는다.
knob 은 `CUDABackend.hash()` 에 들어간다 — 안 넣으면 knob 을 켜고 컴파일한 커널이
끈 뒤에도 캐시에서 계속 나온다 (실제로 테스트가 그 순서 의존성으로 깨졌다).

### wgmma 가 못 하는 형상은 MMAv2 폴백

`supportSparseWGMMA()` 가 M % 64, N % 8, numWarps % 4, 8/16비트, packed K 정합을 본다.
실패하면 `getSparseMMAVersion` 이 remark 를 남기고 2를 돌려준다 — dense 의 `getMMAVersionSafe`
`{3, 2}` 와 같은 idiom 이다. dense 로 도망갈 수 없는 op 이라 "느린 명령 > lowering 실패".

---

## 10. MMAv5 — `tcgen05.mma.sp`

### metadata 가 TMEM 으로 간다

```
tcgen05.mma.sp.cta_group::1.kind::i8 [d-tmem], a-desc, b-desc, [sp-meta-tmem], idesc, pred;
```

`tcgen05.mma` 는 단일 스레드 명령이고 A/B 는 SMEM descriptor, D 와 **metadata 는 TMEM** 이다.
metadata TMEM 레이아웃은 MMA-kind 마다 다르다 — 그것이 v5 의 전부다.

### 8비트 kind 는 항등 매핑이라 기존 기계장치로 끝난다

PTX Figure 266/267 (`.kind::f8f6f4` / `.kind::i8`, M = 128/256) 판독:

```
TMEM lane 0 : [0:0-3][0:4-7][0:8-11][0:12-15] | [0:16-19]...[0:28-31]  ← 32비트 컬럼 c=0
              [0:32-35]...[0:60-63]                                     ← 컬럼 c=1
TMEM lane 1 : 같은 패턴의 A 행 1
...
```

즉 **TMEM lane = A 행**, 32비트 셀 = 인접한 metadata i16 **두 컬럼**(dense 32개).
비트 회계도 정합한다 — M=128, dense K=64: `128 × 64 × 1비트 = 8192 = 128 lane × 2컬럼 × 32비트`.

이것은 정확히 `TensorMemoryEncodingAttr(blockM=128, blockN=metaK, colStride=1)` 이 i16 원소에
대해 기술하는 것이다 (`colStride=1` = packed, `tensorMemoryToLinearLayout` 이 `row→dim0`,
`col→dim1` 항등을 만든다). 그래서 **전용 attr 도, 전용 store lowering 도 필요 없었다**:

```cpp
// SparseBlockedToMMAv5 — dense v5 패턴에 metadata alloc 한 덩어리만 추가
Attribute metaEncoding = TensorMemoryEncodingAttr::get(
    context, /*blockM=*/128, /*blockN=*/metaType.getShape()[1],
    /*colStride=*/1, CGALayout);
auto metaDist = getDefaultLayoutForTmemLdSt(metaMemDescType, numWarps);
Value cvtMeta = ConvertLayoutOp::create(..., metaType.cloneWithEncoding(metaDist), meta);
auto metaTmem = TMEMAllocOp::create(rewriter, loc, metaMemDescType, Type(), cvtMeta);
```

`getDistributedLayoutForTmemLdSt` 의 i16 경로가 `factorMaximalIdentityPrefix(ll, col, dim1, 2)` 로
dim1 방향 packing 을 찾아내 `tcgen05.st.sync.aligned.32x32b.x2.b32` 로 내려간다 (4 warp × 32 lane,
lane 당 2컬럼).

생성 결과 (int8, BLOCK 128×128×128):

```mlir
#tmem  = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>  // 누산기
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4,   colStride = 1>  // metadata
%acc = ttng.tc_gen5_mma %a meta %meta, %b, %acc0, %false, %true ...
```
```
tcgen05.mma.sp.cta_group::1.kind::i8 [ %r561 + 0 ], %rd369, %rd370, [ %r208 + 0 ], %r207, %p12;
tcgen05.mma.sp.cta_group::1.kind::i8 [ %r561 + 0 ], %rd482, %rd483, [ %r208 + 2 ], %r299, %p15;
```

metadata 주소가 명령당 **2 TMEM 컬럼**씩 전진한다 — 한 명령이 dense K=64, 즉 i16 4컬럼
(= 32비트 2컬럼)을 먹기 때문이다. `metaColsPerInst = (2 * mmaSizeK) / 16`.

### instruction descriptor

Table 45 (`.kind::tf32/f16/f8f6f4/i8`) 에는 **K 필드가 없다** — dense K 는 kind + sparsity 로
결정된다. sparse 가 건드리는 것은 두 필드뿐이다.

```cpp
desc.sparsity = 1;          // bit 2
desc.sparsitySelector = 0;  // bits 0-1 — 8비트 kind 는 0 만 유효
```

실제 방출값을 디코드해 확인했다 (`136316068 = 0x82004a4`):
`selector=0, sparsity=1, saturate=0, dtype=2(S32), atype=1(signed8), btype=1, transposeB=1,
N>>3=16(N=128), M>>4=8(M=128), kSize=0`.

> Triton 의 `TCGen5InstructionDescriptor` 는 bit 29 를 `kSize` 로 쓴다. 현재 PTX 문서는 그 비트를
> *"Reserved - Must be 0"* 이라 적고 있다 (2xfp8 K=64 모드용 미문서화 비트로 보인다).
> sparse 와는 겹치지 않으므로 그대로 두었다.

### 16비트 kind 는 비트 하나를 맞바꾼 것이다

Figure 263 (`.kind::f16`, M = 128/256) 을 직접 판독하면, 한 32비트 셀에 **행 m 과 m+8** 이
들어가고 lane 비트 3 이 metadata 컬럼을 고른다:

```
TMEM lane 0 : bits[15:0] = meta[0][0], bits[31:16] = meta[8][0]
TMEM lane 8 : bits[15:0] = meta[0][1], bits[31:16] = meta[8][1]
TMEM lane 16: bits[15:0] = meta[16][0], bits[31:16] = meta[24][0]
```

일반식으로 i16 원소 `(m, j)` 는

```
lane  = 16*(m>>4) + (m&7) + 8*(j&1)      half = (m>>3)&1      32비트 컬럼 = j>>1
```

에 놓인다. 8비트(항등) 배치는 `lane = m, 컬럼 = j>>1, half = j&1` 이므로 **둘의 차이는
`m` 의 비트 3 과 `j` 의 비트 0 을 맞바꾼 것 하나뿐이다.** 비트 회계도 정합한다
(M=128, dense K=32: `128 × 32 × 1비트 = 4096 = 128 lane × 1 컬럼 × 32비트`).
tf32 그림(Figure 265)도 같은 lane 배치라 판독이 교차 확인된다.

스왑 하나이므로 `TensorMemoryEncodingAttr` 에 `sparseMetaRowPaired` 플래그를 하나 붙여
`tensorMemoryToLinearLayout` 에서 `kRow` basis 3 과 `kCol` basis 0 을 바꾸는 것으로 끝난다
(`fp4Padded` 와 같은 성격의 하드웨어 quirk 플래그이고, 기본값이라 기존 IR 출력은 그대로다).

**한 곳만 일반화가 필요했다.** 32비트 셀의 두 half 는 물리적으로 언제나 컬럼 `2c`/`2c+1` 이지만
그것이 *논리적으로* 무엇인지는 레이아웃 사정이다. `getDistributedLayoutForTmemLdSt` 의 i16 경로는
"dim1 방향으로 인접" 만 factor 하고 있었다. 여기서는 그 짝이 dim0 의 +8 이므로,
첫 컬럼 basis 를 **그것이 무엇으로 가든** register basis 로 떼어내는 경로
(`getPackedLayoutForTmemLdSt`)를 마지막 폴백으로 추가했다. 기존 identity-prefix 경로가
성공하는 레이아웃은 거기까지 오지 않으므로 동작이 바뀌지 않는다.
**lowering(`lowerTMemLdSt`)은 손대지 않았다** — 그쪽은 물리 row/col 공간에서 추론하고,
거기서는 어느 쪽이든 짝이 인접한 두 컬럼이기 때문이다.

생성되는 레이아웃 (BLOCK 128×128×64, w8):

```mlir
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1, sparseMetaRowPaired = true>
#linear1 = #ttg.linear<{register = [[8, 0], [0, 2]],
                        lane = [[1, 0], [2, 0], [4, 0], [0, 1], [16, 0]],
                        warp = [[32, 0], [64, 0]], block = []}>
```

register basis `[8,0]` 이 행 짝짓기, lane basis `[0,1]` 이 lane 비트 3 → metadata 컬럼이다.

### sparsity selector 는 64비트 granule 안의 half 다 (실기로 확정)

명령은 metadata 를 **64비트 granule(TMEM 2컬럼) 단위로 주소지정**한다.
8비트 kind 는 granule 을 꽉 채우므로 selector 가 0 이어야 하고(spec 명시), `.kind::f16` 은
32비트만 쓰므로 **연속한 두 명령이 granule 을 공유하고 selector 가 half 를 고른다.**

처음에는 "주소가 그냥 컬럼이고 selector 는 잉여" 라는 해석으로 컬럼 `k` 를 직접 주소지정했는데,
sm_110 이 **`misaligned address` 로 폴트**했다. 즉 granule 정렬이 실재한다. 지금은

```cpp
if (metaHalfGranule) { sparsitySelector = k & 1; metaK = k & ~1; }
```

로 짝수 컬럼을 주소지정하고 selector 로 half 를 고른다. TMEM 할당이 4컬럼 정렬
(`TensorMemoryAllocation.cpp` 의 `columnAlignment = 4`)이라 buffer base 도 안전하다.
descriptor 두 개가 selector 만 1 차이나는 것을 lit 로 고정했다 (`136314900` / `136314901`).

### M = 64 (Layout F) — 구현 완료, 막힌 것은 레이아웃이 아니라 **행 배치**였다

M = 64 는 spec 의 데이터패스 표(§9.7.17.10.5)에서 **Layout F — "4x1, 1/2 datapath utilized",
lane 정렬 0 또는 16** 이다. TMEM 128 lane 중 절반만 쓴다.

> **문서의 M = 64 그림 두 장은 서로 뒤바뀌어 있다.** nibble 하나가 덮는 dense 원소 수는 kind 가
> 정한다 — tf32 는 1:2 라 2개, f16 은 2:4 라 4개. M = 128 쌍은 맞는데(Fig 263 f16 = 4, Fig 265
> tf32 = 2) **M = 64 쌍은 반대다**(Fig 262 "f16" = 2, Fig 264 "tf32" = 4).
> 예전에 "판독이 비트 회계와 맞지 않는다"고 적었던 것이 이것이었다 — 판독이 안 된 게 아니라
> 그림이 바뀐 것이다. v3 에서 "같은 이미지 파일인가"를 확인해 프로브를 아꼈던 것과 같은 확인이다.

Fig 264(실제로는 f16 M = 64)를 읽으면 셀 **내부** 규칙이 M = 128 과 동일하다. 그리고 Triton 의
`TensorMemoryEncodingAttr(blockM = 64)` 가 이미 half-datapath 를 표현한다 — row 비트 4 가
`{0, 0}`(미정의)이 되어 lane 16-31 / 48-63 / … 이 비고, 나머지 row 비트가 행 16-63 을
lane 32-47 / 64-79 / 96-111 로 보낸다. 그 위에 `sparseMetaRowPaired` 스왑을 **그대로** 얹으면
Fig 264 와 일치한다. 즉 인코딩 쪽은 새로 쓸 것이 없었다:

```
dim0 = r0 + 2·r1 + 4·r2 + 16·r5 + 32·r6 + 8·c0      dim1 = r3 + 2·c1
```

**실제로 막고 있던 것은 TMEM 행 배치였다.** `getTmemAllocSizes` 는 row 비트 4 가 `{0,0}` 이면
`nRow /= 2` 로 64행 할당을 만든다. 그러면 할당기가 metadata 를 누산기와 **다른 행 절반**에
자유롭게 놓을 수 있고, 명령은 자기 datapath 쪽 lane 에서 읽으므로 쓰지도 않은 메모리를 읽는다
(실기에서 `misaligned address`). spec 의 alignment restriction 이 말하는 것이 정확히 이것이다:

> The layouts which utilize only half the datapath lanes … Layout F and Layout C,
> **must use the same alignment across matrices A, D and the sparsity metadata matrix.**

Triton 에는 이미 같은 제약 장치가 있었다 — `TensorMemoryAllocation.cpp` 가 TMEM 상주 A 피연산자에
대해 `rowIdConstraints.joinOps(lhsAlloc, accAlloc)` 로 "A 와 누산기는 같은 행" 을 강제한다.
metadata 만 그 그룹에 빠져 있었고, 넣어주자 **bit-exact 로 통과**했다.

```cpp
if (auto mmaOp = dyn_cast<TCGen5MMAOp>(op))
  if (Value aMeta = mmaOp.getAMeta())
    if (getTmemAllocSizes(cast<ttg::MemDescType>(aMeta.getType())).numRows == 64)
      // metadata 를 누산기와 같은 행 그룹에 묶는다
```

게이트는 dense 규칙(`supportMMA`)과 같게 맞췄다 — `M % 64 == 0` 이고 warp 4 또는 8.
M = 32 이나 warp 1/2 는 `SparseBlockedToMMA` 가 `mma.sp` 로 받는다.

**검증**: sm_110 에서 fp16 / bf16 / e4m3 / e5m2 각각 M = 64 형상 3~5개가 bit-exact,
`tcgen05.mma.sp.kind::{f16,f8f6f4}` 생성 확인. 행 16-63 이 lane 32-47 / 64-79 / 96-111 로
간다는 것은 그림에 없는 부분이었는데, **bit-exact 가 그 추정을 판정해 주었다** (틀렸다면 깨진다).

---

## 검증 상태

- 빌드 clang 19.1.7 `-Werror` 에러 0
- lit **290/290** — sparse 관련 12개
  (`accelerate-matmul.mlir` 7: v2 fp16/int8, sm_120, sm_90 fp16/fp8, sm_90 폴백, sm_100 fp8 +
  sm_100 fp16 미변환 / `tritongpu_to_llvm.mlir` 5: v2 4종 + v3 /
  `tritongpu_to_llvm_blackwell.mlir` 1: v5 / `nvgpu_to_llvm.mlir` 1: wgmma.sp PTX)
- pytest: `test_compile_errors.py` 13 + `test_core.py` dot_sparse 24(+10 skip) +
  `test_compile_only.py` 6 (크로스 타깃 명령 선택)
- **크로스 타깃 AOT**: sm_86/90/100/120 각각에 대해 sparse matmul 커널이 기대 명령을 내고
  ptxas 를 통과한다 (수치를 제외한 전 파이프라인)
- 수치 실기: fp16 7/7 (RTX 3060 sm_86) + fp16/bf16/int8/e4m3/e5m2/혼합 fp8 각 8/8
  **비트 단위 일치** (RTX 4070 SUPER sm_89, MMAv2) / 48-48 (RTX 5080 sm_120, MMAv2)
- **MMAv5 실기 검증 완료 — Jetson Thor sm_110.** `verify_dot_sparse.py` **54/54 bit-exact**,
  `tcgen05.mma.sp.kind::f16` 12케이스 + `kind::f8f6f4` 12케이스 포함.
  즉 v5 의 8비트·16비트 metadata TMEM 레이아웃이 둘 다 **하드웨어로 확인**되었다
  (판독이 틀렸다면 bit-exact 가 나올 수 없다). `kind::i8` 도 6케이스 포함 — upstream 의
  `supportsI8Tcgen05MMA()` 가 `cc == 100` 이라 Thor 를 빼고 있었는데, ptxas 가 sm_110a 는
  어셈블하고 sm_103a 만 거부하는 것을 보고 열어서 실기로 확인했다 (dense int8 도 2배 빨라진다).

sm_90 (MMAv3) 의 수치 검증만 기기 확보 후로 남았다.
