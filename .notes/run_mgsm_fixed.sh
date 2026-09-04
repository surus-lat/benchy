#!/usr/bin/env bash
# Re-run the fixed MGSM subtask for all 5 models (50 samples each).
set -u
cd /Users/dobleefe/benchy
set -a; source ~/.hermes/.env 2>/dev/null; set +a

run_model () {
  local model="$1" tag="$2"
  # fresh run dir: mgsm_<tag> (resume-safe: different run_id per model)
  .venv/bin/benchy eval --provider together --model-name "$model" \
    --tasks spanish.mgsm_direct_es_spanish_bench --limit 50 \
    --run-id "mgsm_fixed_${tag}" --exit-policy relaxed \
    > ".notes/mgsm_${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $model mgsm done (exit $?)"
}

run_model "moonshotai/Kimi-K3" "kimik3" &
run_model "zai-org/GLM-5.3" "glm53" &
run_model "zai-org/GLM-5.2" "glm52" &
run_model "deepseek-ai/DeepSeek-V4-Flash-0731" "dsv4flash" &
run_model "MiniMaxAI/MiniMax-M3" "minimaxm3" &
wait
echo "ALL MGSM DONE"