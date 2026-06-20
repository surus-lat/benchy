#!/usr/bin/env bash
# Build the per-model venvs benchy needs.
#
# Two venvs ship today because two model families have incompatible
# transformers pins:
#   .venv      — default. Whisper, Canary (NeMo), Qwen3-ASR (qwen-asr pkg).
#                Pins transformers<5.0.
#   .venv-vox  — Voxtral, plus Whisper + Canary as a bonus (they work on
#                both transformers tracks). Uses transformers from git main.
#
# Idempotent: re-running upgrades each venv to current requirements
# without destroying existing weight caches.
#
# Usage:
#   bash scripts/setup-venvs.sh          # both venvs
#   bash scripts/setup-venvs.sh default  # just .venv
#   bash scripts/setup-venvs.sh vox      # just .venv-vox
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
}

build_default() {
  echo "==> Building .venv (Whisper + Canary + Qwen3-ASR)"
  if [ ! -d .venv ]; then
    uv venv .venv --python 3.12
  fi
  VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install -e '.[transcription,dev]'
  VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install 'nemo-toolkit[asr]>=2.7.0'
  # qwen-asr will downgrade transformers to <5.0, which is what we want
  # in this venv.
  VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install 'qwen-asr>=0.0.6'
  echo "==> .venv ready"
  .venv/bin/python -c "import transformers, qwen_asr, nemo; print('  transformers:', transformers.__version__); print('  qwen_asr:', getattr(qwen_asr, '__version__', 'ok')); print('  nemo:', nemo.__version__)"
}

build_vox() {
  echo "==> Building .venv-vox (Whisper + Canary + Voxtral)"
  if [ ! -d .venv-vox ]; then
    uv venv .venv-vox --python 3.12
  fi
  VIRTUAL_ENV="$REPO_ROOT/.venv-vox" uv pip install -e '.[transcription,dev]'
  VIRTUAL_ENV="$REPO_ROOT/.venv-vox" uv pip install 'nemo-toolkit[asr]>=2.7.0'
  # Voxtral requires transformers >= 5.13 (currently from git main).
  # Install AFTER nemo so it wins the version race.
  VIRTUAL_ENV="$REPO_ROOT/.venv-vox" uv pip install --upgrade \
    'git+https://github.com/huggingface/transformers.git' \
    'mistral-common>=1.11.0'
  echo "==> .venv-vox ready"
  .venv-vox/bin/python -c "import transformers, nemo; print('  transformers:', transformers.__version__); print('  nemo:', nemo.__version__)"
}

require_uv
case "$TARGET" in
  all)
    build_default
    build_vox
    ;;
  default)
    build_default
    ;;
  vox)
    build_vox
    ;;
  *)
    echo "Usage: $0 [all|default|vox]" >&2
    exit 1
    ;;
esac

cat <<EOF

==> Done.

To run a benchmark:
  .venv/bin/benchy eval -c whisper-tiny-transformers ...        # Whisper
  .venv/bin/benchy eval -c qwen3-asr-0.6b-transformers ...      # Qwen3-ASR
  .venv/bin/benchy eval -c canary-1b-flash-transformers ...     # Canary
  .venv-vox/bin/benchy eval -c voxtral-mini-4b-transformers ... # Voxtral

Model YAMLs with a 'venv:' field will pre-flight check and fail fast
with a hint if you launch from the wrong one.
EOF
