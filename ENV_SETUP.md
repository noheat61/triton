# Triton 개발 환경 (dot-sparse-nvidia)

두 머신에서 재현됨. 구축 2026-08-14(sm_86) / 2026-08-18(sm_89).

| | 머신 A | 머신 B (fp8 검증용) |
|---|---|---|
| GPU | RTX 3060 12GB (**sm_86**) | RTX 4070 SUPER (**sm_89**) |
| OS | WSL2, Ubuntu glibc 2.39 | Ubuntu 22.04.5, glibc 2.35 (네이티브) |
| CPU / RAM | 12코어 Zen2 / **15 GB** | 16코어 / 30 GB |
| 드라이버 | 610.43.02 (KMD 610.47, UMD 13.3) | 580.173.02 (CUDA 13.0) |
| conda | miniconda3 26.5.3 | miniconda3 25.5.1 |

sm_89 가 필요한 이유: ptxas 가 `m16n8k64` 의 fp8 변종을 sm_89 이상에서만 받는다.

**CUDA 툴킷은 설치하지 않았다.** 런타임은 `libcuda.so.1` 만 dlopen 하고(`driver.py:16`),
ptxas·cuobjdump·nvdisasm·cudart·cupti 와 LLVM 은 첫 빌드 때 redist 에서 자동으로 받아 `~/.triton/` 에 둔다.
필요한 건 C++17 툴체인뿐이지만 `-Werror` 가 무조건 켜지므로(`CMakeLists.txt:285`) clang 19 로 고정한다.

## 재현

```bash
conda create -y -n triton --override-channels -c conda-forge \
    python=3.12 'clangxx_linux-64=19' 'gxx_linux-64>=11' 'cmake>=3.20,<4.0' ninja lld \
    zlib libxml2 libcurl ncurses libffi
conda activate triton
pip install torch numpy pytest pytest-xdist
pip install -r python/requirements.txt        # lit + nanobind==2.10.2 핀

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang"    # gcc activation script 가 덮어씀
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang++"
MAX_JOBS=3 pip install -e . -v                # 40분~1시간, 첫 빌드는 네트워크 필요
```

이후 증분 빌드: `ninja -C build/cmake.linux-x86_64-cpython-3.12 -j3`. Python 만 고쳤으면 불필요.
빌드 디렉터리는 약 3.9 GB.

**`MAX_JOBS=3` 은 필수.** `setup.py:306` 기본값이 `2 * cpu_count()` 이고 번역 단위 하나가 ~2 GB 라
15 GB 머신에서 즉시 OOM 이다. 30 GB 머신에서도 그대로 썼다.

**`cmake<4.0` 도 필수.** CMake 4 는 3.5 미만 `cmake_minimum_required` 를 거부해 서드파티가 깨진다.

## 버전

conda (아래는 머신 B 실측, 괄호는 머신 A 와 다른 값):

| 패키지 | 버전 | 비고 |
|---|---|---|
| python | 3.12.13 | 3.13/3.14 는 nanobind·torch 호환 회피 |
| clang / clangxx_linux-64 | 19.1.7 / 19.1 | **실제 빌드에 쓰는 컴파일러** |
| gxx_linux-64 | 15.2.0 (A: 16.1.0) | 미사용, 의존성으로 존재 |
| cmake | 3.31.10 (A: 3.31.8) | `<4.0` 필수 |
| ninja | 1.13.0 (A: 1.13.2) | |
| lld | 22.1.8 | libtriton 링크 시간 단축 |
| binutils_linux-64 | 2.46.1 | |
| sysroot_linux-64 | 2.34 (A: 2.39) | |
| zlib / libxml2 / libcurl / ncurses / libffi | 1.3.2 / 2.15.3 / 8.21.0 / 6.6 / 3.7.0 | prebuilt LLVM 링크용 |

pip (직접 설치한 것만; 나머지는 의존성):

| 패키지 | 버전 | 용도 |
|---|---|---|
| torch | 2.13.0 | 테스트·튜토리얼 레퍼런스, `torch.Tensor` 인터롭 |
| numpy | 2.5.2 | 인터프리터 모드, 수치 검증 |
| pytest / pytest-xdist | 9.1.1 / 3.8.0 | `python/test/` 실행, `-n auto` |
| nanobind | 2.10.2 | Python 바인딩 빌드 (`python/requirements.txt` 핀) |
| lit | 18.1.8 | lit 테스트 (conda 환경에 없어 별도 설치 필요) |

빌드 산출물: `triton 3.8.0+git<sha>` (editable install).

torch 2.13.0 이 끌고 오는 CUDA 휠 — 시스템 CUDA 가 필요 없는 이유:

```
cuda-toolkit 13.0.3.0 (cublas, cudart, cufft, cufile, cupti, curand, cusolver,
                       cusparse, nvjitlink, nvrtc, nvtx)
cuda-bindings 13.3.1, nvidia-cudnn-cu13 9.20.0.48, nvidia-cusparselt-cu13 0.8.1,
nvidia-nccl-cu13 2.29.7, nvidia-nvshmem-cu13 3.4.5
```

첫 빌드가 `~/.triton/` 로 받는 것 (핀: `cmake/llvm-build-info.json`,
`cmake/nvidia-toolchain-version.json`):

| | 버전 |
|---|---|
| prebuilt LLVM | `llvm-b010a18d-ubuntu-x64-1` (핀 해시 `941a04e6…`) — clang-24, FileCheck 포함 |
| ptxas | 12.9.86 / **13.3.33 (blackwell)** |
| cuobjdump / nvdisasm | 13.1.80 (구버전 12.8.55 도 함께 존재) |
| cudart / cudacrt | 13.1.80 |
| cupti | 12.8.90 / 13.3.35 (blackwell) |

sm_100a·sm_120a 명령을 어셈블해 확인할 때는 blackwell ptxas 를 직접 쓰면 된다:
`~/.triton/nvidia/nvcc-blackwell/cuda_nvcc-linux-x86_64-13.3.33-archive/bin/ptxas`.

빌드 결과 CMake 설정(검증용): `CMAKE_BUILD_TYPE=TritonRelBuildWithAsserts`,
`CMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang++`,
`CMAKE_LINKER=$CONDA_PREFIX/bin/ld.lld`, `TRITON_CODEGEN_BACKENDS=nvidia;amd`,
`TRITON_BUILD_{PYTHON_MODULE,PROTON,UT}=ON`, `TRITON_CACHE_PATH=~/.triton`.
`ccache` 는 안 깔았다 — `TRITON_BUILD_WITH_CCACHE` 는 기본 ON 이지만 바이너리가 없으면 무시된다.

## 검증

```bash
python -c "import triton, torch; print(triton.__version__, torch.cuda.get_device_capability())"
env -u PYTHONPATH pytest -q python/test/unit/language/test_core.py \
    python/test/unit/language/test_compile_errors.py -k dot_sparse
cd build/cmake.linux-x86_64-cpython-3.12 && \
    ninja triton-opt && lit -q test/TritonGPU/accelerate-matmul.mlir test/Conversion/tritongpu_to_llvm.mlir
```

lit 은 GPU 가 필요 없다. 다른 arch 로 크로스 컴파일도 된다 —
`triton.compile(..., GPUTarget("cuda", 89, 32))` 가 sm_86 머신에서 cubin 까지 만든다(실행만 불가).

## 새 GPU 에서 수치 검증하기 (sm_90 / sm_100 / sm_120)

전체 빌드는 40분~1시간이라 검증할 때마다 하기엔 비싸다. 두 경로가 있고, **타깃 머신 크기로 갈린다.**

### 경로 A — wheel 을 만들어 옮긴다 (타깃이 작거나 여러 대일 때)

`bdist_wheel` 은 `get_cmake_dir()` = 기존 `build/` 를 그대로 재사용하므로 **증분**이다.
`MAX_JOBS` 는 여기서도 필요하다 — `setup.py:306` 의 기본값이 `2 * cpu_count()` 라 그냥 돌리면
증분이라도 여러 TU 가 동시에 뜨는 순간 15 GB 머신이 OOM 이다.

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-clang++"
MAX_JOBS=3 python setup.py bdist_wheel   # 실측 2분 15초(증분), dist/triton-*-cp312-*.whl 216 MB
```

> **타깃 스크립트에는 `MAX_JOBS` 가 없다. 의도한 것이다.** `MAX_JOBS` 를 읽는 곳은
> `setup.py:306` 한 군데뿐이고 그건 소스 빌드 경로다. 타깃은 prebuilt wheel 만 설치하므로
> (실행 로그에 소스 빌드 0건, `tar.gz` 0개) 컴파일이 없다. 런타임에 컴파일되는 C 런처 stub 은
> `runtime/build.py` 가 파일 하나를 직렬로 처리하고 `MAX_JOBS` 를 보지 않는다.
> 소스 빌드를 하는 "경로 B" 에서만 다시 등장한다.

타깃에서는 스크립트 하나로 끝난다:

```bash
scp dist/triton-*.whl dot-sparse-scripts/{setup_target_conda.sh,verify_dot_sparse.py} gpubox:~/
ssh gpubox 'bash setup_target_conda.sh triton-*.whl'
```

`setup_target_conda.sh` 가 하는 일:

1. **preflight** — wheel 이름에서 `cp3XX` 태그를 뽑고, wheel 안 `libtriton.so` 의 glibc 하한을
   호스트 glibc 와 비교하고, GPU·드라이버를 확인한다. 여기서 죽으면 아무것도 건드리지 않았다.
2. **conda** — 없으면 miniconda 설치, wheel 태그가 요구하는 python 으로 env 생성
   (+ `libstdcxx-ng`), **C 컴파일러 확보**.
3. **packages** — torch 먼저, 그 다음 wheel 을 `--force-reinstall --no-deps` 로 덮는다.
4. **library wiring** — stale RPATH 를 이 env 로 재지정하고 libstdc++ 해석을 검증한다.
5. **smoke test** — `tl.dot_sparse` 존재 확인(= 우리 triton 이 활성인지) + 커널 1개 실행
   (= 런처 stub 컴파일 확인) + 이 arch 가 지원하는 dtype 출력.
6. **verify** — `verify_dot_sparse.py`.

`--check-only`(preflight 만) / `--no-verify`(설치만) / `--env NAME` / `--torch SPEC` /
`--torch-index https://download.pytorch.org/whl/cu130`.
실측: 완전히 빈 conda 상태에서 **1분 41초**, sm_86 에서 30/30 bit-exact, 실패 시 exit 1.

**런타임에 C 컴파일러가 필요하다.** Triton 은 커널 시그니처마다 작은 C 런처 stub 을 그때그때
컴파일한다(`runtime/build.py:_find_compiler` → `$CC` 또는 PATH 의 `gcc`/`clang`). python + torch
만 있는 env 로는 커널이 안 돌고 `Failed to find C compiler` 가 뜬다 — 클라우드 이미지에
`build-essential` 이 없는 경우가 흔하다. 스크립트가 시스템 컴파일러를 먼저 찾고, 없으면
`gcc_linux-64` 를 env 에 깔아 `$CC` 를 잡아준다 (root 불필요).

**설치 순서가 왜 중요한가.** torch 는 `triton==3.7.1` 을 **정확히 핀**하므로 한 줄로 같이
설치하면 pip 이 `ResolutionImpossible` 로 죽는다 (실측). torch 를 먼저 넣고 우리 wheel 을
나중에 `--no-deps` 로 덮어야 한다. 손으로 하면:

```bash
conda create -y -n triton-sparse --override-channels -c conda-forge \
    python=3.12 libstdcxx-ng gcc_linux-64   # --override-channels 없으면 ToS 프롬프트에 막힌다
conda activate triton-sparse                       # gcc 패키지가 $CC 를 잡아준다
pip install torch numpy                            # 먼저. triton 3.7.1 이 딸려 들어온다
pip install --force-reinstall --no-deps triton-3.8.0+git*-cp312-cp312-linux_x86_64.whl
pip install patchelf && patchelf --set-rpath "$CONDA_PREFIX/lib" \
    "$(python -c 'import pathlib,triton;print(pathlib.Path(triton.__file__).parent/"_C"/"libtriton.so")')"
python verify_dot_sparse.py
```

`--no-deps` 로 잃는 것은 없다 — 우리 wheel 의 `Requires-Dist` 는
`importlib-metadata; python_version < "3.10"` 하나뿐이라 3.12 에서는 아무 의존성도 없다.
`pip check` 가 *"torch has requirement triton==3.7.1, but you have triton 3.8.0+git..."* 라고
투덜대지만 **기능상 무해**하다 — torch 는 inductor 경로에서만 triton 을 쓰고 버전을 런타임에
검사하지 않는다. 이 저장소의 개발 env 자체가 torch 2.13.0 + triton 3.8.0+git 조합으로
dense dot 747개를 통과한다.

> 나중에 `pip install <다른 패키지>` 가 torch 를 건드리면 pip 이 triton 을 3.7.1 로
> "복구"해 버릴 수 있다. 그게 싫으면 wheel 쪽 버전을 torch 의 핀에 맞춰 버리면 된다 —
> `setup.py:577` 의 `"3.8.0"` 을 `"3.7.1"` 로 바꿔 재빌드(2분)하면
> `3.7.1+git<hash>` 가 되고, PEP 440 은 `==3.7.1` 이 local 라벨을 무시하도록 정해 놓았으므로
> (`Version("3.7.1+git...") in SpecifierSet("==3.7.1")` → True, 실측) 한 줄 설치가 통한다.
> 실제 커밋은 `+git<hash>` 로 남아 모호하지 않다. **커밋하지 말 것.**

wheel 이 **ptxas / ptxas-blackwell / cuobjdump / nvdisasm 을 전부 품고 있다** (`backends/nvidia/bin/`).
`~/.triton` 다운로드(2.9 GB — LLVM 1.7 GB + nvidia redist 777 MB)가 필요 없고, LLVM 은 빌드 전용이라
타깃에는 아예 안 들어간다.

### RPATH 함정 — 검증할 때 속기 쉽다

`libtriton.so` 에는 **빌드 머신 conda env 의 절대 경로**가 RPATH 로 박힌다:

```
$ objdump -p python/triton/_C/libtriton.so | grep RPATH
  RPATH   /home/noheat/miniconda3/envs/triton/lib
```

그래서 이 머신에서 "conda 없이 되는지" 테스트하면 **거짓 통과**한다 — venv 로 돌려도 RPATH 를
타고 conda 의 libstdc++/libz 를 가져다 쓴다(`ldd` 로 확인). 타깃 머신에는 그 경로가 없어
로더가 조용히 시스템 라이브러리로 넘어간다. 제대로 검증하려면 RPATH 를 무력화해야 한다:

```bash
patchelf --set-rpath /nonexistent <venv>/.../triton/_C/libtriton.so
ldd ... | grep libstdc++      # → /lib/x86_64-linux-gnu/libstdc++.so.6 로 넘어가야 정상
```

이 상태로 sm_90/sm_100 크로스 컴파일이 도는 것을 확인했다 (24.04 시스템 libstdc++ 3.4.33 ≥
필요한 3.4.30).

이 함정에 실제로 걸렸다 — 새 conda env 로 스크립트를 돌렸는데 **다른 env 의 libstdc++** 를
가져다 쓰고 있었고 체크는 "ok" 였다. 그래서 `setup_target_conda.sh` 는 RPATH 를 **무조건**
`$CONDA_PREFIX/lib` 로 재지정한 뒤 검증한다. RPATH 에서 못 찾은 항목은 그대로 일반 검색 경로로
넘어가므로 잃는 것은 없고, 어느 머신에서든 해석 결과가 같아진다.

**제약 세 가지:**

| 제약 | 이유 / 확인 방법 |
|---|---|
| **Python 3.12 고정** | nanobind 확장이라 `cp312-cp312` 태그(abi3 아님). pip 이 3.11/3.13 에서는 설치 자체를 거부한다. Ubuntu 24.04 시스템 python 이 3.12.3 이라 그냥 맞는다. 다른 python 을 쓰려면 그 python 으로 재빌드 — `build/` 이름에 python 버전이 들어가므로 **전체 빌드**가 된다 |
| **glibc ≥ 2.38** | 이 머신(WSL, sysroot 2.39)에서 빌드하면 `__isoc23_strtol` 하나 때문에 2.38 을 요구한다. Ubuntu 24.04 ✓ / **22.04(2.35) ✗** |
| libstdc++ ≥ GLIBCXX_3.4.30 | Ubuntu 22.04 의 시스템 libstdc++ 가 정확히 3.4.30 이라 통과. 20.04 는 안 됨 |
| torch 버전 | 기능상 요구는 fp8 dtype 뿐(`torch>=2.1`). 다만 sm_100 은 torch 자체가 cu128+ 빌드여야 한다. **버전 핀 문제는 위 설치 순서로 해결** |
| 드라이버 | Triton 은 cubin 을 직접 로드한다. wheel 이 품은 ptxas 는 sm_90 이하 **12.9.86**, sm_100+ **13.3.33** 이라 CUDA 13 세대 드라이버가 필요하다. B200/H100 호스트면 자동으로 충족 — `nvidia-smi` 로 한 번만 확인 |

확인:

```bash
objdump -T python/triton/_C/libtriton.so | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | tail -1
```

**22.04 타깃을 쓸 거라면** 빌드 env 의 sysroot 를 낮춰 한 번 다시 빌드하면 된다
(`conda install -n triton sysroot_linux-64=2.28` → 전체 재빌드 1회). `__isoc23_strtol` 은
glibc 2.38 *헤더*가 `strtol` 을 리다이렉트해서 생기는 것이고 실제 의존이 아니다.
나머지 심볼 상한은 `GLIBC_2.34`(dlopen/pthread_* 가 libc 로 합쳐진 버전)라 22.04 에서 문제없다.

### 경로 B — 타깃에서 그냥 빌드한다 (클라우드 대형 인스턴스일 때)

`MAX_JOBS=3` 은 **이 머신의 15 GB RAM 제약**이다. H100/B200 호스트는 보통 100+ 코어라
그 제약이 없고, 그러면 빌드가 6~10분으로 떨어져 wheel 전송과 비슷해진다. ABI 걱정도 없다.

```bash
# 필요 없는 백엔드를 빼면 387개 TU 중 137개(AMD 73 + Proton 64)가 사라진다
sed -i 's/BackendInstaller.copy(\["nvidia", "amd"\])/BackendInstaller.copy(["nvidia"])/' setup.py
TRITON_BUILD_PROTON=0 MAX_JOBS=32 pip install -e .
```

`TRITON_BUILD_PROTON=0` 은 wheel 에서 cupti 정적 라이브러리 ~180 MB 도 같이 없앤다
(216 MB → 대략 120 MB). `setup.py:388` 의 백엔드 목록은 env var 가 없어 직접 고쳐야 하고,
**커밋하지 말 것.**

### 무엇을 돌리는가

`dot-sparse-scripts/verify_dot_sparse.py` 는 저장소·pytest 없이 `triton` + `torch` 만으로 돈다.
**정확성 표 + 배속 표를 같이 낸다** — bit-exact 는 레이아웃이 맞다는 뜻이지 빨라졌다는 뜻이 아니라서,
같은 GEMM 을 `tl.dot_sparse` 와 `tl.dot` 으로 **같은 config 스윕**을 돌려 비율을 낸다.
`--speed-size 0` 으로 끄고, `--speed-size 4096` 으로 키우고, `--speed-verbose` 로 config 별
수치를 볼 수 있다. 8비트는 K-contiguous B 를 쓴다 (사용자가 실제로 골라야 하는 레이아웃).

> **num_warps 를 반드시 스윕에 넣어라.** dense 8비트 커널은 `128x128x128` 에서 w4 가 7 TF/s,
> w8 이 58 TF/s — **8배**다 (RTX 3060). 처음에 128x128 타일을 w4 로만 재서 dense int8 이 9 TF/s 로
> 나왔고, 배속이 10x 로 부풀었다. dense 열이 torch 열보다 크게 낮으면 그 배속은 믿지 말 것.
곱이 누산기에서 정확히 표현되는 값만 쓰므로 **불일치 = 레이아웃 오류**(반올림이 아니다)이고,
**실제로 선택된 명령을 같이 출력**한다 — v3/v5 를 검증하려 했는데 조용히 MMAv2 로 폴백한 경우를
잡기 위한 것이다. sm_86 실측: 30/30 bit-exact.

```
[PASS] int8        256x256x512    block 128x256x128    w8  bit-exact   mma.sp.sync.aligned.m16n8k64...
instructions actually exercised:
    9x  mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32
```

shape 목록에 `M >= 128, num_warps >= 4` 케이스가 들어있는 이유가 그것이다 — 작은 shape 만 돌리면
sm_90 은 wgmma 대신 MMAv2 폴백을, sm_100 은 프론트엔드 거부를 보게 된다.

타깃에서 pytest 까지 돌리고 싶으면 저장소를 clone 만 하면 된다(빌드 없이).
`pip install pytest` 후 `pytest -q python/test/unit/language/test_core.py -k dot_sparse`.

## 함정

| 증상 | 원인 / 해결 |
|---|---|
| 빌드 중 OOM / 머신 멈춤 | `MAX_JOBS=3` 누락 |
| `cmake_minimum_required` 에러 | CMake 4.x → `cmake>=3.20,<4.0` |
| `-Werror` 로 모르는 경고가 에러 | 컴파일러가 gcc 로 잡힌 것. `CMakeCache.txt` 의 `CMAKE_CXX_COMPILER` 확인. 툴체인 바꿨으면 `build/` 삭제 후 재빌드 |
| configure 가 `ZLIB::ZLIB` 에서 죽음 | zlib 누락. 컴파일 진입 전이라 원인을 오인하기 쉽다 |
| pytest 가 `No module named 'yaml'` | `/opt/ros/humble` 이 `PYTHONPATH` 로 python3.10 site-packages 를 주입 → ROS pytest 플러그인이 로드됨. 모든 명령을 `env -u PYTHONPATH` 로 (머신 B) |
| LLVM 다운로드 40분 뒤 `NotADirectoryError: compile_commands.json` | 다른 머신에서 복사돼 온 **일반 파일**이 `setup.py:376` 의 symlink 생성을 막는다. `rm compile_commands.json` (gitignored) |
| pip 설치 후 `ninja` 가 `rebuilding 'build.ninja': subcommand failed` | pip 격리 빌드가 CMakeCache 에 `/tmp/pip-build-env-*/` 경로를 박음. `cmake -S . -B build/cmake.linux-x86_64-cpython-3.12 -DCMAKE_MAKE_PROGRAM=$CONDA_PREFIX/bin/ninja` 로 한 번 재configure |
| `libcuda.so.1` not found (WSL) | `/usr/lib/wsl/lib` 가 `LD_LIBRARY_PATH` 에 있는지 확인 |
| **wheel 을 만든 뒤 `ninja` 결과가 반영되지 않음** | `setup.py bdist_wheel` 이 CMake 를 재configure 해 `CMAKE_LIBRARY_OUTPUT_DIRECTORY` 를 `build/lib.linux-x86_64-cpython-312/triton/_C` 로 바꾼다. 이후 `ninja` 는 거기에 링크하고 editable install 이 보는 `python/triton/_C/libtriton.so` 는 그대로 남는다. `.o` 는 새로 컴파일되는데 동작만 안 바뀌어서 원인을 오인하기 쉽다. `cp build/lib.linux-x86_64-cpython-312/triton/_C/libtriton.so python/triton/_C/` 로 복사하거나 `pip install -e .` 를 다시 돌린다 |
| `ninja` 를 `\| grep ... \| head` 로 파이프하면 조용히 죽음 | `head` 가 파이프를 닫아 SIGPIPE. **링크 직전에 죽으면 오브젝트는 새것인데 `.so` 는 옛것**이 된다. 출력은 파일로 리다이렉트하고 나중에 grep |
| 옮긴 wheel 이 `GLIBC_2.38 not found` | 위 "경로 A" 의 sysroot 제약. 타깃 OS 를 올리거나 낮은 sysroot 로 재빌드 |
| 옮긴 wheel 이 torch 의 triton 을 덮지 않음 | torch 는 자체 `pytorch-triton` 을 딸려 온다. `pip install --force-reinstall --no-deps <wheel>` 로 마지막에 덮어쓴다 |
| 새 GPU 에서 sparse 테스트가 전부 skip | `get_supported_sparse_dot_dtypes` 게이트. sm_100 은 fp16/bf16 미지원(의도) |

저장소를 다른 머신에서 복사해 오면 `build/` 의 CMakeCache 가 이전 머신 경로와 LLVM 핀을 가리킨다.
지우고 새로 빌드하는 편이 빠르다.

프로파일링·클럭 고정은 root 가 필요하다: `sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2400,2400`
(즉시 적용, `-rgc` 로 해제), ncu 는 `/etc/modprobe.d` 에 `NVreg_RestrictProfilingToAdminUsers=0` + 재부팅.
