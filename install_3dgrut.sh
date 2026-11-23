#!/usr/bin/env bash
# Install NVIDIA 3DGRUT (3DGUT + MCMC + PPISP, without NHT) into the "aria" conda environment.
#
# Usage: ./install_3dgrut.sh [3DGRUT_DIR] [CONDA_ENV]
#   3DGRUT_DIR  clone location (default: ../3dgrut next to this repository)
#   CONDA_ENV   conda environment (default: aria, created from environment.yml)
#
# Toolchain (override with environment variables):
#   CUDA_HOME             CUDA 12.8 toolkit matching PyTorch cu128 (default: /usr, Lambda Stack)
#   CC / CXX              host compiler supported by CUDA 12.8 (default: gcc-11 / g++-11)
#   TORCH_CUDA_ARCH_LIST  GPU architecture (default: 8.9 = RTX 4090)
#
# Kaolin (playground only) and tiny-cuda-nn (NHT only) are not installed.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRUT_DIR=$(realpath -m "${1:-$SCRIPT_DIR/../3dgrut}")
ENV_NAME=${2:-aria}
SLANGC_VERSION=2026.5.2

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PREFIX="$CONDA_BASE/envs/$ENV_NAME"
PY="$PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "Conda environment not found: $PREFIX (create it with environment.yml)"; exit 1; }

export CUDA_HOME=${CUDA_HOME:-/usr}
export CC=${CC:-/usr/bin/gcc-11}
export CXX=${CXX:-/usr/bin/g++-11}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
"$CUDA_HOME/bin/nvcc" --version | grep -q "release 12.8" || { echo "CUDA_HOME=$CUDA_HOME is not CUDA 12.8"; exit 1; }

echo "[1/5] Cloning 3DGRUT to $GRUT_DIR"
if [[ ! -d "$GRUT_DIR/.git" ]]; then
    git clone --recursive https://github.com/nv-tlabs/3dgrut.git "$GRUT_DIR"
else
    git -C "$GRUT_DIR" submodule update --init --recursive
fi

echo "[2/5] Installing 3DGRUT (keeping the PyTorch/numpy versions of the environment)"
CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
"$PY" -m pip freeze | grep -iE "^(torch|torchvision|xformers|numpy|opencv-python|usd-core|pycolmap|projectaria-tools)==" > "$CONSTRAINTS"
"$PY" -m pip install -e "$GRUT_DIR" -c "$CONSTRAINTS" --extra-index-url https://download.pytorch.org/whl/cu128

echo "[3/5] Building PPISP and fused-ssim"
"$PY" -m pip install --no-build-isolation -c "$CONSTRAINTS" -r "$GRUT_DIR/requirements_extra.txt"

echo "[4/5] Installing slangc $SLANGC_VERSION"
if [[ ! -x "$PREFIX/opt/slang-$SLANGC_VERSION/bin/slangc" ]]; then
    TMP=$(mktemp -d)
    curl -sSL -o "$TMP/slang.tar.gz" \
        "https://github.com/shader-slang/slang/releases/download/v$SLANGC_VERSION/slang-$SLANGC_VERSION-linux-x86_64.tar.gz"
    mkdir -p "$PREFIX/opt/slang-$SLANGC_VERSION"
    tar -xzf "$TMP/slang.tar.gz" -C "$PREFIX/opt/slang-$SLANGC_VERSION"
    rm -rf "$TMP"
fi
cat > "$PREFIX/bin/slangc" <<EOF
#!/bin/sh
exec "$PREFIX/opt/slang-$SLANGC_VERSION/bin/slangc" "\$@"
EOF
chmod +x "$PREFIX/bin/slangc"

echo "[5/5] Writing conda activation hooks for the 3DGRUT JIT builds"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/3dgrut.sh" <<EOF
# 3DGRUT JIT builds (written by install_3dgrut.sh)
for _v in CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST; do
  eval "export _3DGRUT_OLD_\${_v}=\"\\\${\${_v}-__unset__}\""
done
export CUDA_HOME=$CUDA_HOME
export CC=$CC
export CXX=$CXX
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
EOF
cat > "$PREFIX/etc/conda/deactivate.d/3dgrut.sh" <<'EOF'
for _v in CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST; do
  eval "_old=\"\${_3DGRUT_OLD_${_v}}\""
  if [ "$_old" = "__unset__" ]; then unset "$_v"; else export "$_v=$_old"; fi
  unset "_3DGRUT_OLD_${_v}"
done
unset _v _old
EOF

"$PY" -c "import torch, threedgrut, ppisp; from fused_ssim import fused_ssim; print('3DGRUT ready, CUDA:', torch.cuda.is_available(), '| PPISP', ppisp.__version__)"
"$PREFIX/bin/slangc" -version
