#!/usr/bin/env bash
# Run ON the vast.ai box (Ampere, CUDA 12.x devel image) to verify the smedjan CUDA backend.
# Clones the PUBLIC mirror (the box can't reach the Forgejo tunnel), builds --features cuda, and
# runs the full test sweep serially. Writes logs + completion sentinels to /root so the Mac can
# poll over the flaky SSH link. Usage:  nohup bash cuda_verify_remote.sh >/root/sweep.out 2>&1 &
set -u
cd /root
export DEBIAN_FRONTEND=noninteractive
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export PATH="$CUDA_PATH/bin:$HOME/.cargo/bin:$PATH"

echo "=== env ===" > /root/cuda_env.log
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader >> /root/cuda_env.log 2>&1
nvcc --version >> /root/cuda_env.log 2>&1
echo "CUDA_PATH=$CUDA_PATH" >> /root/cuda_env.log

# Toolchain + deps
command -v git >/dev/null || apt-get update -y >/dev/null 2>&1
apt-get install -y git build-essential curl pkg-config >/dev/null 2>&1
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1
fi
. "$HOME/.cargo/env" 2>/dev/null || true

# Repo (public GitHub mirror of Forgejo origin)
if [ ! -d smedjan ]; then
  git clone --depth 1 https://github.com/smedjan/smedjan.git >> /root/cuda_env.log 2>&1
fi
cd smedjan
git rev-parse HEAD >> /root/cuda_env.log 2>&1

# Build the CUDA backend
echo "BUILD_START" > /root/cuda_build.log
CARGO_INCREMENTAL=0 cargo build --release --no-default-features --features cuda >> /root/cuda_build.log 2>&1
echo "BUILD_EXIT=$?" >> /root/cuda_build.log

# Full test sweep — CUDA shares device state, so run serially with ignored tests included.
echo "TEST_START" > /root/cuda_test.log
CARGO_INCREMENTAL=0 cargo test --release --no-default-features --features cuda -- \
  --include-ignored --test-threads=1 >> /root/cuda_test.log 2>&1
echo "TEST_EXIT=$?" >> /root/cuda_test.log

# Sentinels for polling from the Mac
grep -E "test result:" /root/cuda_test.log | tail -5 > /root/cuda_summary.log 2>&1
grep -E "FAILED| failed|panicked|error\[" /root/cuda_test.log | head -40 >> /root/cuda_summary.log 2>&1
echo "SWEEP_DONE" >> /root/cuda_test.log
