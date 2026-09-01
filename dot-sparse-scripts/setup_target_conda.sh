#!/usr/bin/env bash
# Set up a conda env on a fresh GPU box and verify tl.dot_sparse numerically.
#
# Installs no compiler and builds nothing: it consumes a Triton wheel built
# elsewhere (`python setup.py bdist_wheel`, ~2 min incremental) together with a
# pip torch. Intended for sm_90 / sm_100 / sm_120 machines this branch has not
# been validated on yet.
#
#   scp dist/triton-*.whl dot-sparse-scripts/*.{sh,py} gpubox:~/
#   ssh gpubox 'bash setup_target_conda.sh triton-*.whl'
#
# Options:
#   --env NAME          conda env name (default: triton-sparse)
#   --torch SPEC        torch requirement (default: torch)
#   --torch-index URL   extra index, e.g. https://download.pytorch.org/whl/cu130
#   --check-only        run the preflight checks and stop
#   --no-verify         install but skip the numerical run
set -euo pipefail

WHEEL=""
ENV_NAME="triton-sparse"
TORCH_SPEC="torch"
TORCH_INDEX=""
CHECK_ONLY=0
RUN_VERIFY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --torch) TORCH_SPEC="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --no-verify) RUN_VERIFY=0; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) WHEEL="$1"; shift ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------- preflight

say "preflight"

[[ -n "$WHEEL" ]] || die "no wheel given: $0 <triton-*.whl>"
WHEEL="$(readlink -f "$WHEEL")"
[[ -f "$WHEEL" ]] || die "wheel not found: $WHEEL"
ok "wheel $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

# The wheel is an ABI-specific CPython extension (nanobind, not abi3), so the
# env's python must match its tag exactly -- pip refuses otherwise.
PY_TAG="$(basename "$WHEEL" | grep -oE 'cp3[0-9]+' | awk 'NR==1')"
[[ -n "$PY_TAG" ]] || die "cannot read a cp3XX tag out of the wheel name"
PY_VER="3.${PY_TAG#cp3}"
ok "wheel needs python $PY_VER (tag $PY_TAG)"

# libtriton.so is built against the build machine's glibc. One symbol
# (__isoc23_strtol) pulls a 2.38 floor when built on a glibc-2.38+ host.
# The heredoc lives in a function: a heredoc inside $( ) is not portable.
wheel_glibc_floor() {
    python3 - "$1" <<'PY'
import pathlib, re, subprocess, sys, tempfile, zipfile
z = zipfile.ZipFile(sys.argv[1])
name = next(n for n in z.namelist() if n.endswith("_C/libtriton.so"))
with tempfile.TemporaryDirectory() as d:
    so = pathlib.Path(d, "libtriton.so")
    so.write_bytes(z.read(name))
    try:
        out = subprocess.run(["objdump", "-T", str(so)], capture_output=True,
                             text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit(1)
    vs = sorted(int(m.split(".")[1]) for m in set(re.findall(r"GLIBC_2\.[0-9]+", out)))
    if vs:
        print(f"2.{vs[-1]}")
PY
}

HOST_GLIBC="$(ldd --version | awk 'NR==1' | grep -oE '[0-9]+\.[0-9]+$')"
NEED_GLIBC="$(wheel_glibc_floor "$WHEEL" 2>/dev/null || true)"
if [[ -n "$NEED_GLIBC" ]]; then
    if [[ "$(printf '%s\n%s\n' "$NEED_GLIBC" "$HOST_GLIBC" | sort -V | awk 'NR==1')" == "$NEED_GLIBC" ]]; then
        ok "glibc $HOST_GLIBC >= $NEED_GLIBC required by the wheel"
    else
        die "glibc $HOST_GLIBC < $NEED_GLIBC required by the wheel.
        Rebuild it against an older sysroot on the build machine:
          conda install -n <build-env> sysroot_linux-64=2.28   # then a full rebuild"
    fi
else
    warn "could not read the wheel's glibc floor (objdump missing?) -- continuing"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY="$HERE/verify_dot_sparse.py"
if [[ "$RUN_VERIFY" == 1 ]]; then
    [[ -f "$VERIFY" ]] || die "verify_dot_sparse.py must sit next to this script ($HERE).
        Copy it over too, or pass --no-verify to install only."
    ok "verify_dot_sparse.py found"
fi

command -v nvidia-smi >/dev/null || die "nvidia-smi not found; is this a GPU box?"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1')"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR==1')"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1' | tr -d '.')"
ok "gpu $GPU (sm_$CC), driver $DRIVER"
# Triton loads cubins directly; the bundled ptxas for sm_100+ is CUDA 13.x, so
# the driver has to be from that generation.
if [[ "$CC" -ge 100 ]] && [[ "${DRIVER%%.*}" -lt 580 ]]; then
    warn "sm_$CC uses the CUDA 13 ptxas; driver $DRIVER may be too old (>= 580 expected)"
fi

[[ "$CHECK_ONLY" == 1 ]] && { say "check-only: stopping here"; exit 0; }

# --------------------------------------------------------------------- conda

say "conda"

if ! command -v conda >/dev/null && [[ ! -x "$HOME/miniconda3/bin/conda" ]]; then
    warn "conda not found, installing miniconda to ~/miniconda3"
    curl -fsSL -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm -f /tmp/miniconda.sh
fi
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
ok "conda base $CONDA_BASE"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ok "env $ENV_NAME exists, reusing"
else
    # libstdcxx-ng so the RPATH step below always has a new-enough libstdc++ to
    # point at, whatever the host distro ships.
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge \
        "python=$PY_VER" libstdcxx-ng >/dev/null
    ok "env $ENV_NAME created (python $PY_VER)"
fi
conda activate "$ENV_NAME"
ACTUAL_PY="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$ACTUAL_PY" == "$PY_VER" ]] || die "env python is $ACTUAL_PY but the wheel needs $PY_VER"
ok "python $(python -V 2>&1 | cut -d' ' -f2) at $(command -v python)"

# Triton compiles a small C launcher stub for every kernel signature at run
# time (runtime/build.py:_find_compiler wants $CC, or gcc/clang on PATH), so a
# python-only env is not enough. Many cloud images ship no build-essential.
if [[ -n "${CC:-}" ]] && command -v "$CC" >/dev/null; then
    ok "C compiler \$CC=$CC"
elif command -v gcc >/dev/null; then
    ok "C compiler $(command -v gcc)"
elif command -v clang >/dev/null; then
    ok "C compiler $(command -v clang)"
else
    warn "no C compiler found, installing gcc_linux-64 into the env"
    conda install -y -n "$ENV_NAME" --override-channels -c conda-forge \
        gcc_linux-64 >/dev/null
    # The compiler package sets $CC through its activation script.
    conda deactivate && conda activate "$ENV_NAME"
    [[ -n "${CC:-}" ]] && command -v "$CC" >/dev/null \
        || die "gcc_linux-64 installed but \$CC is still unset"
    ok "C compiler \$CC=$CC"
fi

# ------------------------------------------------------------------ packages

say "packages"

# torch first, on purpose. torch pins triton exactly (e.g. triton==3.7.1), so
# installing both in one command fails with ResolutionImpossible; installing
# ours afterwards with --no-deps overrides the pinned one. Nothing is lost:
# this wheel's only Requires-Dist is importlib-metadata for python < 3.10.
PIP_ARGS=(--quiet)
[[ -n "$TORCH_INDEX" ]] && PIP_ARGS+=(--extra-index-url "$TORCH_INDEX")
python -m pip install --upgrade pip "${PIP_ARGS[@]}" >/dev/null
python -m pip install "${PIP_ARGS[@]}" "$TORCH_SPEC" numpy
ok "torch $(python -c 'import torch; print(torch.__version__)')"

python -m pip install --quiet --force-reinstall --no-deps "$WHEEL"
ok "triton $(python -c 'import triton; print(triton.__version__)') (overrode torch's pin)"

# ------------------------------------------------------------ library wiring

say "library wiring"

# The wheel carries an absolute RPATH from the *build* machine's conda env. If a
# directory happens to exist at that path here, the loader silently prefers it,
# which is how a broken setup can look healthy (or a healthy one break). Repoint
# it at this env unconditionally so the resolution is the same everywhere;
# entries not found under RPATH still fall through to the normal search path, so
# this only ever adds determinism.
LIBTRITON="$(python -c 'import pathlib, triton; print(pathlib.Path(triton.__file__).parent / "_C" / "libtriton.so")')"
OLD_RPATH="$(objdump -p "$LIBTRITON" 2>/dev/null | awk '/RPATH|RUNPATH/ {print $2}' | awk 'NR==1' || true)"
[[ -n "$OLD_RPATH" ]] && ok "wheel RPATH was $OLD_RPATH (build machine's path)"

python -m pip install --quiet patchelf
patchelf --set-rpath "$CONDA_PREFIX/lib" "$LIBTRITON"
ok "RPATH -> \$CONDA_PREFIX/lib"

NEED_CXX="$(objdump -T "$LIBTRITON" 2>/dev/null | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | awk 'END{print}' || true)"
RESOLVED="$(ldd "$LIBTRITON" 2>/dev/null | awk '/libstdc\+\+/ {print $3}')"
[[ -n "$RESOLVED" ]] || die "libstdc++ did not resolve for $LIBTRITON -- run: ldd $LIBTRITON"
case "$RESOLVED" in
    "$CONDA_PREFIX"/*|/lib/*|/usr/lib/*) ;;
    *) die "libstdc++ resolved from an unexpected place: $RESOLVED
        That is a leftover RPATH or LD_LIBRARY_PATH; unset it and re-run." ;;
esac
HAVE_CXX="$(strings "$RESOLVED" | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | awk 'END{print}')"
if [[ -z "$NEED_CXX" ]] || [[ "$(printf '%s\n%s\n' "$NEED_CXX" "$HAVE_CXX" | sort -V | awk 'NR==1')" == "$NEED_CXX" ]]; then
    ok "libstdc++ $RESOLVED provides ${HAVE_CXX:-?} (needs ${NEED_CXX:-?})"
else
    die "libstdc++ $RESOLVED provides only $HAVE_CXX but $NEED_CXX is needed.
        conda install -n $ENV_NAME -c conda-forge libstdcxx-ng"
fi

# ------------------------------------------------------------------- smoke

say "smoke test"

python - <<'PY'
import sys
import triton
import triton.language as tl
# The decisive check that *our* triton is active and not torch's pinned one:
# tl.dot_sparse only exists on this branch.
if not hasattr(tl, "dot_sparse"):
    sys.exit("tl.dot_sparse is missing -- the pinned triton is still installed.\n"
             "Re-run: pip install --force-reinstall --no-deps <wheel>")
print(f"  triton {triton.__version__} has tl.dot_sparse")
import torch
major, minor = torch.cuda.get_device_capability()
cc = major * 10 + minor
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_supported_sparse_dot_dtypes
supported = get_supported_sparse_dot_dtypes(GPUTarget("cuda", cc, 32))
names = [d for d in ("float16", "bfloat16", "int8", "float8e4nv", "float8e5")
         if supported(getattr(tl, d))]
print(f"  sm_{cc} supported sparse dtypes: {', '.join(names) or '(none)'}")
if not names:
    if 90 <= cc < 100:
        sys.exit(f"sm_{cc}: the MMAv3 sparse path ships disabled because its numerics have\n"
                 "  never run on a Hopper device. Re-run this script with\n"
                 "    TRITON_ENABLE_UNVERIFIED_SPARSE_WGMMA=1 bash setup_target_conda.sh <wheel>\n"
                 "  to close that out.")
    sys.exit(f"sm_{cc} has no supported sparse dtype -- nothing to verify")
PY

# Launch one trivial kernel: this is what forces the C launcher stub to be
# compiled, which is a separate failure mode from everything above. It has to
# live in a real file -- @triton.jit needs inspect.getsource, so a heredoc fed
# to `python -` fails with "should be defined in a Python file".
SMOKE="$(mktemp --suffix=.py)"
trap 'rm -f "$SMOKE"' EXIT
cat > "$SMOKE" <<'PY'
import torch
import triton
import triton.language as tl


@triton.jit
def _touch(p):
    tl.store(p + tl.arange(0, 4), tl.arange(0, 4))


x = torch.zeros(4, dtype=torch.int32, device="cuda")
_touch[(1, )](x)
assert x.tolist() == [0, 1, 2, 3], x.tolist()
print("  launcher stub compiles and a kernel runs")
PY
python "$SMOKE"
ok "smoke test passed"

# ------------------------------------------------------------------- verify

if [[ "$RUN_VERIFY" == 1 ]]; then
    say "numerical verification"
    # Do not let `set -e` swallow the verdict: run it, keep the status, and say
    # plainly whether this chip is good before exiting with it.
    set +e
    python "$VERIFY"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
        printf '\n\033[42;30m  PASS  \033[0m \033[1mcorrectness and speed are both good on this GPU\033[0m\n'
    else
        printf '\n\033[41;37m  FAIL  \033[0m \033[1msomething is wrong -- read the verdict block above\033[0m\n'
        echo   "        A speed FAIL means sparse is slower than dense: suspect the"
        echo   "        instruction spelling or a rate-limited path, not tuning."
        echo   "        Re-run one dtype with --speed-verbose to see every config."
    fi
    exit $rc
else
    say "done (verification skipped)"
    echo "  conda activate $ENV_NAME && python verify_dot_sparse.py"
fi
