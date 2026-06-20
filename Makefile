# benchy convenience targets — wraps scripts/setup-venvs.sh and the
# per-model `benchy eval` commands. Pre-flight venv check enforces the
# right binary; targets below pre-bind it so you can't get it wrong.

.PHONY: help setup-venvs setup-venv-default setup-venv-vox \
        smoke-whisper smoke-canary smoke-qwen smoke-voxtral \
        test test-all

help:
	@echo "benchy — useful targets:"
	@echo
	@echo "  make setup-venvs        build both .venv and .venv-vox"
	@echo "  make setup-venv-default build just .venv (Whisper + Canary + Qwen3-ASR)"
	@echo "  make setup-venv-vox     build just .venv-vox (Voxtral)"
	@echo
	@echo "  make smoke-whisper      1-sample FLEURS es smoke on whisper-tiny"
	@echo "  make smoke-canary       1-sample on Canary-1b-flash"
	@echo "  make smoke-qwen         1-sample on Qwen3-ASR-0.6B"
	@echo "  make smoke-voxtral      1-sample on Voxtral-Mini-4B (uses .venv-vox)"
	@echo
	@echo "  make test               unit tests (default venv)"

setup-venvs:
	bash scripts/setup-venvs.sh all

setup-venv-default:
	bash scripts/setup-venvs.sh default

setup-venv-vox:
	bash scripts/setup-venvs.sh vox

smoke-whisper:
	.venv/bin/benchy eval -c whisper-tiny-transformers \
	  --tasks transcription.fleurs_es_latam \
	  --limit 1 --log-samples \
	  --run-id make_smoke_whisper --exit-policy smoke

smoke-canary:
	.venv/bin/benchy eval -c canary-1b-flash-transformers \
	  --tasks transcription.fleurs_es_latam \
	  --limit 1 --log-samples \
	  --run-id make_smoke_canary --exit-policy smoke

smoke-qwen:
	.venv/bin/benchy eval -c qwen3-asr-0.6b-transformers \
	  --tasks transcription.fleurs_es_latam \
	  --limit 1 --log-samples \
	  --run-id make_smoke_qwen --exit-policy smoke

smoke-voxtral:
	.venv-vox/bin/benchy eval -c voxtral-mini-4b-transformers \
	  --tasks transcription.fleurs_es_latam \
	  --limit 1 --log-samples \
	  --run-id make_smoke_voxtral --exit-policy smoke

test:
	.venv/bin/python -m pytest -q --no-header

# Run unit tests under BOTH venvs to catch divergence. Useful before commits.
test-all: test
	.venv-vox/bin/python -m pytest -q --no-header
