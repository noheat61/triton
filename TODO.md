# `tl.dot_sparse` (2:4 structured sparsity) — NVIDIA 구현 TODO

브랜치 `dot-sparse-nvidia`, base `087e97245` (upstream/main). 원본 보존 `backup/sparse24-old`.
갱신: 2026-09-01. 환경은 `ENV_SETUP.md`, 코드 해설은 `DOT_SPARSE_INTERNALS.md`.

## 현재 상태 — sm_80~sm_120 전 세대 구현 (MMAv2/v3/v5)

| 세대 | 명령 | 아키텍처 | dtype | 상태 |
|---|---|---|---|---|
| MMAv2 | `mma.sp.sync` | sm_80–89 | fp16/bf16/int8/fp8(89+) | **실기 검증 완료** |
| MMAv2 | `mma.sp::ordered_metadata` | sm_120+ | 위와 동일 | **실기 검증 + 성능 완료** (RTX 5080) |
| MMAv3 | `wgmma.mma_async.sp` | sm_90 | 위와 동일 | 구현 완료, **기본 비활성** (실기 미검증) |
| MMAv5 | `tcgen05.mma.sp` | sm_100–119 | fp16/bf16/fp8, int8(cc==100), M % 64 == 0 | **실기 검증 완료** (Thor sm_110) |

- 검증 (GPU 없이 가능한 범위 전부): clang 19 `-Werror` 0, lit **298/298**, C++ unit **280/280**,
  pytest sparse **40** + compile-error 13 + cross-target 26, sm_86/90/100/120 타깃 AOT 컴파일에서
  각 세대의 기대 명령 생성 + **ptxas 통과**
- 수치 실기 검증: sm_86 / sm_89 / sm_120 (MMAv2) / **sm_110 (MMAv5, 54/54 bit-exact)**.
  sm_90 (v3) 는 기기 확보 후
- **PR 미제출**

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

### GPU 없이 검증하는 방법 (v3/v5 작업의 전제)

`triton.compile(ASTSource(...), target=GPUTarget("cuda", cc, 32))` 은 해당 아키텍처가 없어도
TTIR→TTGIR→LLVM→PTX→**cubin** 까지 전부 돈다 (ptxas 는 `~/.triton/nvidia/nvcc*/bin/ptxas`,
sm_100+ 는 blackwell 빌드 13.3.33 자동 선택). 즉 **수치를 뺀 전 파이프라인이 검증 가능**하다.
`test_compile_only.py` 의 `_compile_sparse_matmul` 이 그 형태이고, sm_80/86/90/100/120 각각에서
어떤 sparse 명령이 선택되는지를 회귀 테스트로 고정해 두었다.

PTX 문법·아키텍처 지원 여부만 볼 때는 ptxas 를 직접 때리는 것이 가장 빠르다
(`.version`/`.target` 만 바꿔 한 줄 어셈블). v3/v5 의 shape·dtype·selector 유효 범위를 이 방법으로
전수 확인했다.

---

## 확정된 결정 (v3/v5 를 제약하므로 유지)

**사용자 계약: metadata 는 아키텍처와 무관하게 `[..., M, K_dense/16]` i16.** 각 i16 이 4비트 그룹 4개,
각 그룹은 dense 4원소 중 남긴 2개의 2비트 인덱스. 아키텍처 차이는 전부 컴파일러 내부로 숨긴다 —
**세 세대 전부 검증됨**: v2 레지스터(`LinearEncoding`+`ConvertLayout`), v3 같은 레이아웃에
warp basis 순서만 M-우선, v5 TMEM 상주(`tensor_memory_encoding`+`tmem_alloc`).
사용자 커널은 세 아키텍처에서 **한 글자도 다르지 않다** (`12-dot-sparse.py` 가 그대로 돈다).
#6714 의 *"Metadata layouts are user-defined and can vary between backends"* 와 반대 방향이고,
v2/v3/v5 를 한 PR 로 올리면 그 주장은 유지 불가다. `DotSparseOp::verify()` / `semantic.dot_sparse` 의
meta 검사(i16, `K/8`)가 이 계약이고, 이제 `verifySparseDotMetadata()` 로 공유해
`WarpGroupDotOp` / `TCGen5MMAOp` 도 같은 검사를 통과해야 한다.

**새 IR attr 없음 — v5 까지 지켜졌다.** `SparseMetadataEncodingAttr` 를 만들었다가 제거(`1aeeed9af`).
`LinearEncodingAttr` 는 broadcast(zero) basis 를 허용하고 전단사는 그것을 제거한 뒤에만
요구하므로(`TritonGPUAttrDefs.td:675-690`), warp 내 중복과 N-warp 복제가 그대로 표현된다.
리팩터 전후 수치가 비트 단위로 동일했다.
v3 도 같은 attr 을 쓰고, v5 는 **기존 `tensor_memory_encoding`(packed, colStride=1)** 으로
8비트 metadata TMEM 레이아웃이 정확히 표현된다 (scales 처럼 전용 attr 을 만들 필요가 없었다).
새 op 도 없다 — `WarpGroupDotOp`/`TCGen5MMAOp` 에 optional 피연산자를 붙였고,
덕분에 파이프라이너·barrier·accumulator 패스를 손대지 않았다.

**`sp::ordered_metadata` 를 sm_100+ 에서 채택 (이전 결정 뒤집음).** 니모닉만 바꿔도 비트 패킹이
같아 수치가 동일하다(vLLM marlin_24 PR #5136 도 니모닉만 바꿈). 예전엔 sm_86/89 에서 측정 이득이
0 이고 PTX 8.5 하한이 생긴다는 이유로 미채택했는데, **ptxas advisory 가 문자 그대로였다.**

> Modifier `.sp::ordered_metadata` should be used ... it is expected to have
> **substantially reduced performance on some future architectures**

RTX 5080 (sm_120) 2048³ 실측, dense-equivalent TF/s:

| | `mma.sp` | `sp::ordered_metadata` | 개선 |
|---|---|---|---|
| fp16 | 35.8 | **143.1** | **4.00x** |
| fp16 (f16 누산) | 36.9 | **205.3** | **5.56x** |
| int8 | 58.3 | **357.2** | **6.13x** |
| bf16 | 143.5 | 143.5 | 1.00x |
| fp8 e4m3 / e5m2 | 261.0 | 261.0 | 1.00x |

**벌칙이 선택적이다** — fp16 과 int8 만 4~6배를 물고 있었고 bf16/fp8 은 이미 빠른 경로였다.
고치기 전엔 sparse fp16/int8 이 **같은 칩의 dense 보다 느렸다**(0.44x, 0.23x). 지금은 1.74x / 1.43x 이고
둘 다 cuBLAS·`torch._int_mm` 을 1.4배 이긴다. 수치는 48/48 bit-exact 유지(예상대로).

허용 nibble 이 `{0100,1000,1100,1001,1101,0110,1110}` 로 제한되는데 `compress()` 가 인덱스를
증가 순서로 채우므로 생성 가능한 6개 값이 전부 그 안이다 — 그래서 인코딩이 안 바뀐다.

**sm_80–89 는 legacy 유지** (측정 이득 0, PTX 하한 올릴 이유 없음).
**MMAv3 는 해당 없음** — ptxas 가 `wgmma.mma_async.sp::ordered_metadata` 를
*illegal modifier* 로 거부하고, plain `wgmma.mma_async.sp` 에는 advisory 를 아예 안 낸다.
즉 이 문제는 `mma.sync`(MMAv2) 전용이다.

**SM 게이트 `[80, 130)`, 단 fp16/bf16 은 `[100, 120)` 제외.** 이전엔 하한만 있어 sm_90+ 에서도
MMAv2 로 매치되어 경고 없이 dense 보다 느려졌다. 이제 세대별로 명령을 골라주므로 전 범위를 열었다.
유일한 예외가 sm_100~103 의 16비트 sparse (v5 metadata TMEM packing 미표현, 아래 MMAv5 절 참조).

---

## 성능 (RTX 4070 SUPER sm_89, 4096³, k-major B, 튜닝 후)

**루프라인은 명령별로 직접 재야 한다.** "8비트 = fp16 의 2배" 추정은 틀렸다.
레지스터 상주 피연산자로 MMA 만 반복해 측정한 명령 처리율(`dot-sparse-scripts/mma_tput.py`,
dense-equivalent TF/s):

| 명령 | f32/s32 누산 | f16 누산 |
|---|---|---|
| fp16 `m16n8k16` dense | 70.1 | 152.8 |
| fp16 sparse `m16n8k32` | 161.1 | 321.3 |
| int8 dense / sparse `m16n8k64` | 305.7 / **642.7** | — |
| fp8 dense `m16n8k32` | 157.2 | 322.5 |
| fp8 sparse `m16n8k64` | **313.7** | **명령 없음** |

- **컨슈머 Ada 는 FP32 누산 MMA 를 절반 속도로 돌린다** (fp16 70 vs 153, fp8 157 vs 323).
  데이터센터 칩에는 없는 제약이고, int8 은 s32 누산이라 벌칙이 없어 상한이 fp8 의 두 배다.
- **sparse fp8 은 f16 누산 변종이 없다** — ptxas: *"F16 accumulator and F8 floating point type
  not supported on .target 'sm_89'"* (dense fp8 은 됨). 이 칩에서 sparse fp8 은 절반 속도에 묶인다.

커널 효율:

| dtype | 커널 | 상한 | 효율 | dense 라이브러리 대비 |
|---|---|---|---|---|
| fp16 sparse | 129~135 TF | 161 | 80~84% | 1.75x (cuBLAS 74) |
| fp8 e4m3 sparse | 233~250 TF | 314 | 80% | 1.65~1.77x (`_scaled_mm` 141) |
| int8 sparse | 296~335 TF | 643 | 46~52% | 5.1x (`_int_mm` 65) |

**fp8 은 느린 게 아니다** — fp16 과 같은 80% 를 내고 상한 자체가 절반이다. 여유가 남은 쪽은 int8.
같은 config 에서 int8/fp8 의 PTX·SASS 명령 수가 완전히 동일하고 opcode 만 다르므로
(`IMMA.SP.16864` vs `QMMA.SP.16864`, LDSM/LDS/STS/LDGSTS 동일, spill 0) 데이터 이동은 더 짤 것이 없다.

### 8비트 피연산자는 K-contiguous 여야 한다 (레이아웃만으로 1.32~1.70x)

`ldmatrix.trans` 는 16비트 단위로만 전치한다. B 가 N-contiguous(`torch.matmul` 기본)면 8비트
dot operand 를 만들 수 없어 SMEM→레지스터 경로가 무너진다 — 루프 1회(mma.sp 64개)당
`ldmatrix` 8 / `ld.shared` **296** / `prmt` **192**, K-contiguous 면 24 / 40 / 0.
같은 폭으로 각각 튜닝한 최적값: fp8 177 → 233 TF(**1.32x**), int8 191 → 324 TF(**1.70x**), fp16 은 무관.

fp8 dense 라이브러리도 같은 제약이다(`torch._scaled_mm` 은 B col-major 강제, cuBLASLt 는 TN 만).
사용자 레벨 해결: B 를 `(N, K)` 로 저장해 전치해 넘긴다(수치는 두 레이아웃 모두 비트 단위 일치 확인).

### f16 누산기 (fp16 입력) — 구현 완료, opt-in

`out_dtype=tl.float16` 일 때만 쓰고 기본은 f32 (dense `tl.dot` 과 같은 규약). bf16·fp8 은 변종이
없어 semantic 에서 막는다. 데이터센터 칩은 f32 누산도 풀 속도라 **컨슈머 전용 최적화**.

| 누산기 | 최적 config | 실측 | 상한 대비 | 상대오차 K=512 / K=4096 |
|---|---|---|---|---|
| f32 | (128,128,32) w4 st4 | 122.7 TF | 76% | 4.8e-07 / 3.4e-06 |
| f16 | (256,128,64) w8 st3 | 172.7 TF | 54% | 8.4e-04 / 2.35e-03 |

**1.41x** (2x 가 아닌 이유: 연산이 2배 빨라지면 데이터 이동이 병목). 오차는 √K 로 자란다.
참고로 f32 누산 결과를 fp16 으로 저장만 해도 2.1e-04 이므로, 출력이 fp16 인 추론 경로면
1e-3 수준을 받아들이고 1.4x 를 사는 셈이다. K 가 긴 학습 리덕션에는 쓰지 말 것.
오버플로는 f16 누산 고유 문제가 아니다(출력이 65504 를 넘는 스케일에서는 f32→fp16 저장도 같이 터짐).

### 블록/스테이지는 8비트에서 dense 와 다르다

8비트는 MMA 하나가 dense K=64 를 먹어 `BLOCK_K=64` 면 K-step 당 MMA 가 1개뿐이고 SMEM 예산이
파이프라인 깊이를 제한한다. 최적: fp8 `(128,128,128) stages=2`, int8 `(128,256,128) w8 stages=3`,
fp16 `(128,128,64) w4 stages=2`. → 튜토리얼에 `triton.autotune` 을 붙이는 것이 맞다 (아직 안 붙임).

### metadata 경로는 병목이 아니다

파이프라이너가 이미 `async_copy_global_to_local` + `local_load`(목적지가 바로 `#linear`)로 처리하고
별도 `convert_layout` SMEM 왕복이 없다. 튜닝된 config 에서 metadata 로드를 루프 밖으로 빼면 **-4%**
(빼는 게 오히려 느림). 튜닝 전 config 의 18~26% 차이는 SMEM 예산 교란이었다.

### 미결 — GROUP_M / 작은 M

L2 CTA 스위즐(GROUP_M 1/4/8/16) 측정이 클럭 변동에 묻혔다(같은 config 이 64~140 TF). 클럭 고정 후
재측정 필요. `batch 1024` 처럼 M 이 작은 형상은 튜닝 후에도 67~76% 라 split-K / 더 큰 BLOCK_N 후보.

---

## 재시도 금지

**Ada 에서 8비트 전치 `ldmatrix` 는 존재하지 않는다.** Triton 코드에 경로는 있으나
(`Utility.cpp:225` 의 `supportLdStMatrixB8()`) `TargetFeatures.h:44` 가 `cc >= 100` 으로 게이트하는
**Blackwell 전용 명령**(`ldmatrix.m16n16.x4.trans.b8`)이다. `movmatrix` 도 b16 뿐이다.
"8비트를 16비트로 보고 전치 후 `prmt` 로 복원" 도 깨진다 — 8비트 B 피연산자는 lane t 가 n=t>>2 하나를
담아 **인접한 두 n 이 4칸 떨어진 lane 에 있는데**, 16비트 단위는 인접한 두 n 바이트라 `ldmatrix.trans`
가 둘을 같은 lane 에 준다. `prmt` 는 lane 내부 연산이라 복구 불가(warp `shfl` 필요).
그래서 업계 표준이 "8비트 텐서코어 GEMM 은 k-major 피연산자". dense `tl.dot` 도 같은 손실을 겪는다
(튜닝 후 fp8 141 vs 136 TF, int8 192 vs 160 TF) → Triton 의 구조적 한계이지 sparse 경로 문제가 아니다.

**uint8 은 범위 밖.** Triton 정수는 signless 라 `uint8`/`int8` 이 IR 에서 구분되지 않는다.
dense int8 dot 도 `.s8` 만 쓴다.

**k64 metadata 매핑은 추정하지 말 것.** 16비트의 확장이 아니었고(첫 구현은 int8 수치가 전부 틀렸다),
프로브로 실측해야 했다 — 각 i16 에 1비트(`0x4444`/`0xEEEE`)를 심고 one-hot B 로 복원 결과를 읽어
6라운드 이진 인코딩 (`dot-sparse-scripts/probe_k64_meta.py`).

> **v3 는 프로브가 필요 없었다** — PTX 문서가 `mma.sp` 와 **같은 그림 파일**을 재사용하므로 매핑이
> 정의상 동일하다. 문서를 뒤져 "같은 이미지인가"를 확인하는 것이 프로브보다 훨씬 값싸다.
> 확인 방법: PTX ISA `index.html` 을 `curl` 로 받아 두 절의 `<img src>` 를 비교
> (`grep -o 'id="[^"]*sparse[^"]*"'` 로 앵커를 찾고 오프셋 주변 `<img>` 를 뽑는다).
> 그림 자체는 `_images/<name>.png` 로 받아 눈으로 읽으면 된다.
>
> **v5 도 같은 방법으로 풀렸다** — TMEM 레이아웃은 v2/v3 와 전혀 다른 그림이지만,
> 8비트(Fig 267)와 16비트(Fig 263) 모두 비트 회계가 정합했고 **sm_110 실기에서 bit-exact 로
> 확인**됐다. M = 64 (Fig 262/264)도 마찬가지인데, 다만 **문서의 두 그림이 서로 뒤바뀌어 있다**
> (nibble 당 dense 원소 수가 kind 와 반대로 붙어 있다). 바꿔 읽으면 M = 128 과 같은 규칙이고,
> 실제 장애물은 TMEM 행 배치였다. 상세는 `DOT_SPARSE_INTERNALS.md` §10 의 M = 64 절.

---

## 남은 일

### 컨슈머 Blackwell (sm_120) — **구현 완료**, MMAv2 코드 그대로

ptxas 13.3.33 확인:

| | sm_89 | sm_90a | sm_100a | sm_120a |
|---|---|---|---|---|
| `mma.sp...m16n8k64...e4m3` | OK | OK | OK | **OK** |
| `mma.sp...m16n8k32...f16` (f16 누산) | OK | OK | OK | **OK** |
| `mma.sp...m16n8k64...e4m3.f16` (fp8 + f16 누산) | 거부 | — | — | **거부** |
| `tcgen05.mma[.sp]` | 거부 | 거부 | OK | **거부** (TMEM 없음) |

RTX 50xx 에는 tcgen05/TMEM 이 없어 `mma.sp.sync` 를 쓴다 (`getMMAVersionSafe` 도 sm_120 에서
dense dot 에 v2 를 고른다) → 우리 구현이 그대로 유효하다. fp8 + f16 누산은 sm_120a 에서도
없으므로 semantic 의 "f16 누산은 fp16 입력만" 게이트가 그대로 맞다.

- [x] SM 게이트를 sm_120 포함으로 확장 (`get_supported_sparse_dot_dtypes`, `SparseBlockedToMMA`,
      `test_core.py` skip 조건). sm_100 은 tcgen05 경로가 있으니 제외 유지
- [x] `[80,90) ∪ [120,130)` 6개 dtype 을 sm_120 타깃으로 AOT 크로스 컴파일 →
      PTX 생성 + ptxas 통과 확인 (GPU 없이 검증 가능한 범위 전부)
- [x] **sm_120 실기 수치 검증 완료** — RTX 5080, driver 595.84, Ubuntu 24.04.
      `verify_dot_sparse.py` **48/48 bit-exact**, 6개 dtype 전부 (fp16/bf16/int8/e4m3/e5m2/f16누산),
      7개 형상 x 1/2/4/8 warp. 선택된 명령도 확인: `m16n8k32`(16비트) / `m16n8k64`(8비트).
      즉 **fp8 sparse 가 소비자 Blackwell 에서 동작**한다 (sm_89 이후 두 번째 확인)
- [x] **sm_120 속도 측정 완료** (RTX 5080, 2048³, dense-equivalent TF/s).
      `sp::ordered_metadata` 적용 후:

      | dtype | sparse | dense | 배속 | torch | 대 torch |
      |---|---|---|---|---|---|
      | fp16 | 143.1 | 82.3 | **1.74x** | 100.5 (cuBLAS) | 1.42x |
      | fp16 (f16 누산) | **205.3** | — | 1.43x (sparse f32 대비) | — | 2.04x |
      | bf16 | 143.5 | 82.3 | **1.74x** | 101.0 | 1.42x |
      | int8 | **357.2** | 250.6 | 1.43x | 252.9 (`_int_mm`) | 1.41x |
      | fp8 e4m3/e5m2 | 261.0 | 159.5 | **1.64x** | 250.8 (`_scaled_mm`) | 1.04x |

      **f32 누산 반감은 sm_120 에도 있다** — f16 누산이 여전히 1.43x 빠르다(sm_86 은 1.60x).
      즉 f16 누산 opt-in 은 소비자 Blackwell 에서도 값어치가 있다.
      int8 이 절대 성능 1위(357)이고 fp8 은 1.64x 로 배속이 가장 크다.
- [ ] 8비트 B 레이아웃 세금이 sm_120 에도 남는지. sm_86 실측 int8 sparse
      k-major 97.7 vs n-major 57.5 = **1.70x** (Ada 실측 1.70x 와 일치).
      `verify_dot_sparse.py` 는 8비트를 항상 k-major 로 재므로, 비교하려면
      `measure_speed` 의 `b.T.contiguous().T` 를 한 번 빼고 돌려보면 된다

**8비트 레이아웃 세금은 sm_120 에서도 남는다 (예상 정정).** `ldmatrix.m16n16.x4.trans.b8` 이
sm_120a 에 존재하고 `TargetFeatures.h:44` 의 게이트도 `cc >= 100` 이라 열려 있는데, 실제로는
쓰이지 않는다 — `lowerLdStMatrix`(`Utility.cpp:225`)의 전치 타일이 요구하는 레지스터 배치가
MMAv2 8비트 dot operand 와 맞지 않아 `divideLeft` 가 실패한다. sm_120 타깃 AOT 컴파일에서
**dense int8 `tl.dot` 도 sparse 와 똑같이** `ld.shared` 160 / `prmt` 96 이 나온다(sm_89 와 동일 수치).
b8 전치 ldmatrix 는 MMAv5 계열 레이아웃용이다. 즉 이건 sparse 경로가 아니라 upstream dense
경로의 미구현이고, 별도로 손대야 할 문제다. → **row-major B 로도 k-major 수준** 기대는 철회.

### MMAv3 (Hopper `wgmma.mma_async.sp`) — **구현 완료**

**metadata 매핑은 프로브가 필요 없었다.** PTX ISA 가 `wgmma.mma_async.sp` 의 metadata 레이아웃
그림으로 **`mma.sp` 와 같은 이미지 파일을 재사용**한다 (`sparse-mma-metadata-16832-f16bf16.png`,
`sparse-mma-metadata-16864-u8s8-{first,last}32col.png` — 문서 HTML 에서 두 절이 같은 `<img src>` 를
가리킨다). 즉 warp 내부 thread↔metadata 대응이 sm_89 에서 실측한 v2 매핑과 **동일**하다.
그림을 직접 판독해 v2 구현의 식과 일치함을 재확인했다 (16비트 `row=T>>2, col=T%2`, `E[31:16]=row+8`;
8비트 `row=(T>>2)+8*(T&1)`, `col=2*((T>>1)&1)`, `E[31:16]=col+1`).

달라지는 것은 **warp basis 순서 하나**뿐이다. warpgroup 은 4 warp 을 M 방향으로 쌓으므로
(`%warpid % 4` 가 A 의 `16*(%warpid%4) .. +15` 행 담당) `NvidiaMmaEncodingAttr` 의 Hopper warp order
`getMatrixOrder(rank, rowMajor=false) = {0,1}` 를 따라 M-warp basis 를 N-warp broadcast 앞에 놓는다.
→ `getSparseMetadataLayout(..., rowMajorWarpOrder)` 파라미터 하나 추가.

- [x] `WarpGroupDotOp` 에 optional `$aMeta` (신규 op 아님) → 파이프라이너·dot-wait·accumulator
      패스가 전부 그대로 동작. `AttrSizedOperandSegments` 는 출력 IR 을 바꾸지 않는다
      (`operandSegmentSizes` 는 ODS 프린터가 elide)
- [x] `mmaVersionToInstrShape(..., isSparse)` — v3 는 instrShape 에 K 가 있고 sparse 는 dense K
      (16비트 k32 / 8비트 k64). lowering 은 A 타일·A descriptor 만 절반씩, B 와
      imprecise-acc 카운트는 dense K 유지
- [x] sparse lhs 는 SMEM 고정. 레지스터 lhs 변종은 존재하지만 그걸 쓸 이유인 in-register
      파이프라이닝(`splitRSDot`)이 metadata 도 K 로 쪼개야 해서 보류 (`splitRSDot` 에 가드 추가)
- [x] wgmma 가 표현 못 하는 형상은 **MMAv2 로 폴백 + remark** (dense 의 `{3,2}` 와 동일 idiom).
      dense 로 도망갈 수 없는 op 이라 "느린 명령 > lowering 실패"
- [x] `max_num_imprecise_acc` 는 `tt.dot_sparse` 에 없으므로(#6714 공용 op) sm_90 dense 기본값
      `1<<30` 을 패스에서 지정. 32 미만이면 fp8+f32 누산이 verifier 에서 거부된다
- [x] 검증: `-Werror` 0, lit 290/290, sm_90 타깃 AOT 컴파일로 6개 dtype 전부
      `wgmma.mma_async.sp` 생성 + ptxas 통과. PTX 피연산자 순서
      (`{D}, a-desc, b-desc, sp-meta, sp-sel, scale-d, 1, 1, trans-a, trans-b`) lit 로 고정
- [x] **기본 비활성화** (`24be349ef`). `knobs.nvidia.enable_unverified_sparse_wgmma`
      (`TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1`) 로만 열린다. knob 을 **백엔드 hash 에 포함**했다 —
      안 하면 knob 켜고 컴파일한 커널이 끈 뒤에도 캐시에서 계속 나온다.
      기본 상태에서 Hopper 사용자는 평범한 "unsupported on this target" 을 받는다
- [ ] Hopper 실기 수치 검증 → **통과하면 knob 제거**.
      `verify_dot_sparse.py` 가 sm_90 에서 knob 없이 실행되면 그 방법을 안내하고 exit 2 를 낸다

  **왜 Blackwell 검증으로 대체할 수 없는가.** `SparseBlockedToMMAv5` 는 v3 코드를 하나도 쓰지
  않는다 — `getSparseMetadataLayout` 0회, `mmaVersionToInstrShape(isSparse)` 0회. metadata 가
  레지스터 vs TMEM, op 이 `warp_group_dot` vs `tc_gen5_mma`, lowering 이 `WGMMA.cpp` vs
  `MMAv5.cpp` 로 전부 갈린다. Thor 가 통과해도 `WGMMA.cpp` 는 한 줄도 실행되지 않는다.

  **H100 없이 이미 닫은 것** (H100 대여가 어려울 때의 대안이 아니라, 리스크 축소):
  - metadata 매핑은 추정이 아니다 — PTX 문서가 `mma.sp` 와 **같은 그림 파일**을 재사용
  - `warpsPerCTA=[4,1]` 레이아웃은 v2 와 바이트 동일 → `block (256,64,K)` w4 로
    **5080/3060 에서 bit-exact 검증됨** (`verify_dot_sparse.py` 가 커버리지 표로 보고)
  - 레이아웃 42케이스를 PTX 그림 공식과 전수 대조, 0 불일치
  - SASS 가 `HGMMA.SP.64x128x32` / `IGMMA.SP.64x128x64` 를 shape·개수까지 일치하게 생성
  - 남은 미검증: N-warp>1 의 broadcast basis 위치, SMEM descriptor, `wgmma.fence` 앞 sp-meta
    조립, 파이프라이너 상호작용, 그리고 **성능** (sm_120 에서 4~6배 함정이 있었다)

### MMAv5 (Blackwell `tcgen05.mma.sp`) — **8비트 구현 완료, 16비트 보류**

`tcgen05.mma.sp.cta_group::1.kind::{f16,f8f6f4,i8} [d-tmem], a-desc, b-desc, [sp-meta-tmem], idesc, pred`
가 sm_100a/sm_103a 에서 어셈블되고 sm_120a 에서 거부된다(확인). 하드웨어 훅
`MMAv5.cpp` 의 `sparsity : 1` / `sparsitySelector : 2` 를 이제 실제로 쓴다.

**핵심: metadata 는 TMEM 상주이고, 레이아웃이 MMA-kind 마다 다르다.** PTX Figure 266/267 (8비트)과
Figure 262/263 (16비트)을 직접 판독한 결과:

| kind | M | TMEM 매핑 | Triton 표현 가능? |
|---|---|---|---|
| `f8f6f4`/`i8` | 128/256 | **항등** — lane = A 행, 32비트 셀 = 인접 metadata 2컬럼(dense 32) | **가능** (packed `tensor_memory_encoding`) |
| `f16` | 128/256 | 한 셀에 **행 m 과 m+8** + selector 로 sub-column 선택 | **불가** (M 방향 packing) |
| 모두 | 64 | half-datapath(Layout F/C), 위와 다른 레이아웃 | 미조사 |

비트 회계로 두 그림 모두 정합함을 확인했다. 8비트 M=128: `128행 × 64 dense × 1비트 = 8192비트
= 128 lane × 2 컬럼 × 32비트`. 16비트 M=128: `128 × 32 × 1 = 4096 = 128 lane × 1 컬럼`.

- [x] `TCGen5MMAOp` 에 optional `$a_meta` (TMEM memdesc) + `verifyDims` 오버라이드 + verifier
- [x] descriptor `sparsity = 1`, `sparsitySelector = 0` (8비트 kind 는 0 만 유효).
      실제 방출값을 비트 단위로 디코드해 확인
- [x] `tcgen05.mma.sp` 방출 — `[sp-meta-tmem]` 은 b-desc 와 idesc 사이
- [x] `SparseBlockedToMMAv5` 패턴. **새 attr 없음**: metadata memdesc 를
      `tensor_memory_encoding<blockM=128, blockN=K_dense/16, colStride=1>` i16 으로 두면
      `getDefaultLayoutForTmemLdSt` → `ConvertLayout` → `tmem_alloc` 기존 경로가 그대로 돌고
      `tcgen05.st.32x32b.x2.b32` 로 i16 2개씩 packed 저장된다
- [x] 검증: sm_100 타깃 AOT 컴파일로 int8/fp8 → `tcgen05.mma.sp.kind::{i8,f8f6f4}` 생성 +
      ptxas 통과. metadata TMEM 오프셋이 명령당 2컬럼씩 전진하는 것까지 lit 로 고정
- [x] **16비트(fp16/bf16) 구현 완료 — `tcgen05.mma.sp.kind::f16`.** 한 32비트 셀에 행 m/m+8 을
      넣는 packing 은 8비트 항등 배치와 **비트 하나(`m` 비트3 ↔ `j` 비트0) 차이**일 뿐이라,
      `TensorMemoryEncodingAttr` 에 `sparseMetaRowPaired` 플래그 하나와
      `getDistributedLayoutForTmemLdSt` 의 i16 packing 일반화로 끝났다. 전용 store lowering 도,
      새 op 도 필요 없었다 (`lowerTMemLdSt` 는 물리 row/col 공간이라 무변경). 상세는
      `DOT_SPARSE_INTERNALS.md` §10
- [x] **selector 는 64비트 granule 안의 half 였다 (실기로 확정).** "주소가 컬럼이고 selector 는
      잉여" 라는 첫 해석으로 홀수 컬럼을 주소지정했더니 sm_110 이 `misaligned address` 로 폴트.
      지금은 짝수 컬럼 + `selector = k & 1`. 8비트는 granule 을 꽉 채우므로 selector 0 그대로
- [x] **M = 64 (Layout F) 구현 완료 — 막고 있던 건 레이아웃이 아니라 TMEM 행 배치였다.**
      `blockM = 64` 인코딩이 이미 half-datapath 를 표현하고 `sparseMetaRowPaired` 스왑이 그 위에
      그대로 얹힌다. 진짜 문제는 `getTmemAllocSizes` 가 64행 할당을 만들면 할당기가 metadata 를
      누산기와 **다른 행 절반**에 놓을 수 있다는 것이었다 (실기 `misaligned address`).
      spec 의 alignment restriction 이 그것이고, Triton 에 이미 있던
      `rowIdConstraints.joinOps` 에 metadata 를 넣어 해결. 게이트는 dense 와 같게
      `M % 64 == 0` + warp 4/8. sm_110 에서 fp16/bf16/e4m3/e5m2 M=64 형상 bit-exact
- [x] 하한은 dtype 이 아니라 **경로**에 붙는다 — M = 32 나 warp 1/2 는 `mma.sp` 로 간다
- [x] **v5 패턴은 dense 와 같은 int8 규칙(`supportsI8Tcgen05MMA()`)을 쓴다.** 전에는 v5 패턴이
      `[100,120)` 전체에서 int8 을 받아 그 명령이 없는 칩에도 낼 수 있었다 — ptxas 는 통과시키므로
      실기에서 틀린 결과나 fault 로만 드러날 버그였다 (`bdde31c60`)
- [x] **upstream 의 그 규칙이 `cc == 100` 이라 Thor 를 빼고 있었고, 그건 틀렸다.**
      ptxas 는 아키텍처를 구별해서 답한다 — `tcgen05.mma.kind::i8` 을 **sm_100a / sm_110a 는
      어셈블하고 sm_103a 는 이름을 대며 거부**한다 (*"Feature '.kind::i8' not supported on
      .target 'sm_103a'"*). Thor 실기로 확인: dense int8 이 `tcgen05.mma.kind::i8` 을 내고
      `torch._int_mm` 과 **결과가 정확히 일치**하며 8192³ 에서 **105.7 → 205.9 TF/s (약 2배)**.
      sparse int8 도 같은 게이트를 타므로 132.5 → 191.5 로 올랐고 54/54 bit-exact 유지.
      `supportsI8Tcgen05MMA()` 를 `cc == 100 || cc == 110` 으로 고쳤다 —
      **이건 sparse 가 아니라 upstream dense 경로의 수정이다.** sm_103 은 그대로 제외

#### Jetson Thor (sm_110) — Blackwell 이지만 sm_100 과 다르다

`sm_101a` 가 PTX 9.0 에서 **`sm_110a` 로 개칭**된 것이고 TMEM/tcgen05 가 있다.
`sm_arch_from_capability(110)` → `sm_110a`, ptxas 는 blackwell 빌드, emit PTX **9.3**
(sm_110a 는 **PTX 9.0+** 필요 — 8.7 은 거부).

| 확인 항목 | 결과 |
|---|---|
| `tcgen05.mma[.sp]` sm_110a | OK, **advisory 없음** (sm_120 의 `.sp` 함정 없음) |
| int8 tcgen05 | **있다** (upstream 게이트가 빼고 있었음, 위 항목 참조) |
| fp8 tcgen05 | OK, SASS `UTCQMMA` 네이티브 |
| TMEM 사용량 | 256 / 512 컬럼 (상한 여유) |
| `supportsReuseB` / `supports2xFp8` / `supports4xFp4` / `ExclusiveTMEMAlloc` | 전부 cc==107 전용 → Thor 는 표준 경로 |
| `requiresFp4Padding` | 110 포함이지만 fp4 전용, 우리 경로 무관 |
| BLOCK_M=256 metadata 주소 | `meta[+0,+2,+4,+6]` — TMEM 이 128 lane 뿐이라 행 128–255 가 컬럼 4–7 로 쌓이는 것이 memdesc 레이아웃으로 정확히 처리됨 |
#### Thor 실측 (2026-09-01) — sparsity 는 **형상이 커야** 값을 한다

`verify_dot_sparse.py` **54/54 bit-exact** (fp16/bf16 은 `kind::f16`, fp8 은 `kind::f8f6f4`).

속도는 fp16, C 를 fp16 으로 저장, dense/sparse 양쪽 다 튜닝, **L2 CTA 스위즐(GROUP_M) 포함**,
그리고 세 커널을 **번갈아** 재서 클럭 드리프트를 제거한 값이다(라운드 4회, 편차 ±0.5%):

| 크기 | sparse | Triton dense | torch(cuBLAS) | sp/dense | sp/torch |
|---|---|---|---|---|---|
| 2048³ | 84.1 | 80.2 | 90.9 | 1.05x | 0.93x |
| 4096³ | 129.7 | 123.1 | 130.5 | 1.05x | 0.99x |
| 8192³ | **148.7** | 99.2 | 110.1 | **1.50x** | **1.35x** |

**fp16 도 2048~4096 에서는 값을 하지 않고, 8192³ 에서야 넘어간다.** 텐서코어가 병목이 아니면
MMA 를 절반으로 줄여도 아무 일이 없기 때문이다. 판정은 `probe_mmav5_metadata.py` 의
**MMA 민감도**(MMA 개수를 절반으로 줄였을 때 런타임이 얼마나 변하는가)로 한다.

| dtype @ 8192³ | sparse/dense | MMA 민감도 | 해석 |
|---|---|---|---|
| fp16 | 1.09~1.50x | 1.18x | 텐서코어가 병목에 걸리기 시작 → sparsity 가 먹힌다 |
| fp8 e4m3 | 0.96x | **1.05x** | 8192 에서도 여전히 대역폭 바운드 → 지렛대 없음 |

**8비트는 어느 크기에서도 값을 하지 않는다.** 연산이 fp16 의 두 배로 빨라 대역폭 벽에 먼저 닿기
때문이고, 구현 결함이 아니라 이 칩의 루프라인이다 (Thor 는 LPDDR5X 통합 메모리라 벽이 낮다).
sparse/dense 가 1.0 아래인데 **민감도가 1.0 을 크게 웃돌면** 그때가 진짜 결함이다 —
Blackwell 에서 legacy `mma.sp` 스펠링을 쓰던 버그가 그 모양이었다.

> **측정 함정 세 개. 전부 이 프로젝트가 한 번씩 걸렸다.**
> 1. **L2 CTA 스위즐 없는 커널로 비교하지 말 것.** 평범한 2D grid 는 8192³ 에서 sparse 50 / dense 38
>    TF/s 로 무너진다. GROUP_M 을 넣으면 146 / 103 이다. 3x 차이라 여기서 나온 배속은 전부 무의미하다.
>    (TODO 의 "GROUP_M 측정이 클럭 변동에 묻혔다" 항목이 이것이었다 — 묻힌 게 아니라 컸다.)
> 2. **dense 기준선을 같이 튜닝할 것, 그리고 번갈아 잴 것.** 좁은 config 목록 + 스위즐 없는 dense +
>    torch 를 맨 마지막에 재는 순서로는 4096³ 에서 "sparse 1.78x" 가 나온다. 같은 하드웨어에서
>    제대로 재면 1.05x 다. 한 번은 이 잘못된 수치를 이 문서에 적었다.
> 3. **단발 스윕으로 config 를 고르지 말 것.** 스윕은 config 마다 한 번씩만 재므로 "빠른 config" 와
>    "운 좋은 순간" 을 구별하지 못한다. 승자 하나만 나중에 정확히 다시 재면 **잘못 고른 것을 정확히
>    재는 것**일 뿐이다. int8 sparse 가 같은 형상에서 실행마다 70 / 129 / 131 TF/s 로 나왔고,
>    fp16 은 이 때문에 8192³ 에서 실측 144 대신 125 로 보고되고 있었다.
>    `verify_dot_sparse.py` 는 이제 스윕 상위 `SPEED_CANDIDATES`(4) 개를 전부 번갈아 재고
>    그 중앙값으로 고른다 — 세 번 연속 실행에서 128.9 / 129.6 / 128.4 로 안정됐다.

- metadata TMEM 저장/배리어의 직렬화 비용은 **지배적이지 않다**. BLOCK_K 를 64→128 로 키워
  iteration 당 저장 횟수를 절반으로 줄여도 sparse/dense 비율은 오히려 내려간다.
  즉 metadata TMEM 다중버퍼링은 지금 우선순위가 아니다
- `sparse` vs `sparse-static-meta` 격차 ~11% 는 metadata 의 **글로벌 트래픽**이다
  (BLOCK_K=128 이면 행당 16바이트짜리 stride 접근이라 섹터 효율이 나쁘다).
  사용자 레벨에서 metadata 를 K-major 로 재배치하면 줄어들 여지가 있다 — 미측정
- ncu 는 이 머신에서 `ERR_NVGPUCTRPERM` (perf counter 권한). 위 진단은 전부 ablation 으로 냈다
- [x] **TMA + warp specialization 경로 확인 완료** (`bench_tma_sparse.py`).
      질문이 둘이었고 답이 다르다:

      1. **되는가 — 된다.** persistent 루프 + tensor descriptor + `warp_specialize=True` 안의
         `tl.dot_sparse` 가 `tcgen05.mma.sp.kind::f16` 으로 내려가고 **오차 0**. metadata 는
         자기 디스크립터를 타고, warp specialization 이나 TMA 기계장치가 그 존재를 알 필요가 없다.
         Blackwell 에서 cuBLAS 급 커널이 쓰는 경로이므로 여기 못 들어가면 막다른 길이었는데 아니다.
         (metadata 디스크립터의 최내차원이 16바이트 이상이어야 해서 **BLOCK_K >= 128** 이 조건)
      2. **더 빠른가 — Thor 에서는 아니다.** 한 프로세스 안에서 번갈아 잰 값:

      | 8192³ | sparse | dense | sparse/dense |
      |---|---|---|---|
      | TMA + warp spec | 113.9 | 87.7 | 1.30x |
      | `tl.load` + L2 스위즐 | **145.8** | 101.3 | 1.44x |
      | torch (cuBLAS) | 95.2 | | |

      TMA 경로가 naive 대비 **0.78x** 다. Thor 는 **SM 이 20개**뿐이라 148-SM 급을 겨냥한
      persistent + warp-spec 설계가 값을 못 한다. sparsity 자체의 이득은 두 경로 모두에 있다
      (1.30x / 1.44x). 최고 sparse 는 145.8 로 **cuBLAS 대비 1.53x**.

      > 단 여기 TMA 커널은 손으로 쓴 것이고 epilogue subtiling·2-CTA·CLC 스케줄링이 없다.
      > "이 형상·이 부품에서 persistent 설계가 값을 못 했다" 로 읽을 것이지 TMA 자체의 판정이 아니다.
      > 데이터센터 Blackwell 에서는 반대로 나올 가능성이 크고, 그건 기기 확보 후

- [x] **async MMA 의 metadata WAR 해저드 수정.** `LowerLoops.cpp` 의 `waitBuffers` 가 A/B/scale 만
      담고 metadata 를 빠뜨려서, 파이프라인된 루프가 in-flight `tcgen05.mma.sp` 가 아직 읽고 있는
      metadata TMEM 버퍼를 다음 iteration 이 덮어쓸 수 있었다. 조용히 틀리는 종류의 버그다
- [ ] `two_ctas` / `is_async` + barrier 경로에서의 metadata 수명 (지금은 동기 1-CTA 만)
- [ ] block-scaled sparse(nvfp4 등, `TCGen5MMAScaledOp`)는 **별도 PR**
- [x] **판독 재확인 완료 — 프로브 대신 실기 bit-exact 로.** 8비트/16비트 TMEM 레이아웃이 둘 다
      sm_110 에서 54/54 bit-exact 다. 레이아웃 판독이 틀렸다면 나올 수 없는 결과이므로
      one-hot B 프로브는 불필요해졌다
- [x] int8 tcgen05 경로도 Thor 에서 검증됨 (`tcgen05.mma.sp.kind::i8` 6케이스 bit-exact)

기대치: 스펙 비율은 어디서나 2x(B200 datasheet FP8 dense 4.5 PF / sparse 9 PF, *"Dense is one-half
of the sparse specification"*)이고 우리가 sm_89 에서 그 83~88% 를 잡았다. 하지만 sparsity 는 연산만
반으로 줄이고 weight 트래픽은 그대로라 **math-bound 형상에서만** 이득이다 — H100 cuSPARSELt 실사용
보고치가 큰 M 에서 1.27~1.41x, 작은 모델 ~1.0x, decode 1.18x. sm_100 은 dense 경로가 TMA +
warp specialization 으로 성숙해 있어 **1.2~1.5x 를 기대치로 잡는 것이 정직하다.**

### PR 제출

- [ ] 첫 줄에 "Based on / extends #6714" + `@SamGinzburg`, `@jcaip` 멘션
- [ ] "AMD(#6714)와 NVIDIA 가 같은 op 을 공유하며 양쪽 다 동작" 을 앞세울 것 —
      백엔드 하나짜리보다 공용 IR op 추가의 정당성이 커진다
- [ ] 커밋은 논리 단위 유지. NVIDIA 백엔드는 core maintainer 소유라 리뷰 표면적을 의식할 것
- [ ] `#6714` / `#6891` 에 코멘트 (NVIDIA 구현 보유 알림) — 판단 대기
- [ ] **타이밍 재검토**: v3/v5 를 기다릴 근거 중 하나였던 "fp8 미검증" 이 해소됐다.
      v2(16비트+8비트, fp8 실기 검증 완료)만으로 완결된 PR 이 되므로 지금 올리고 v3/v5 를
      후속 PR 로 가는 편이 통과 확률이 높을 수 있다

`Co-authored-by: Samuel Ginzburg <ginzburg@meta.com>` 는 `302982abe` 에 이미 적용됨
(`TT_DotSparseOp` 정의가 #6714 와 동일). 라이선스 MIT, CLA/DCO 요구 없음.
