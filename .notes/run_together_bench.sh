#!/usr/bin/env bash
# Benchmark 5 Together serverless models on spanish + structured_extraction.
# Runs sequentially per model (two tasks each), models in parallel via wait.
set -u
cd /Users/dobleefe/benchy
set -a; source ~/.hermes/.env 2>/dev/null; set +a

run_model () {
  local model="$1" tag="$2"
  echo "[$(date +%H:%M:%S)] START $model"
  .venv/bin/benchy eval --provider together --model-name "$model" \
    --tasks spanish --limit 50 \
    --run-id "together_${tag}_full" --exit-policy relaxed \
    > ".notes/bench_${tag}_spanish.log" 2>&1
  echo "[$(date +%H:%M:%S)] $model spanish done (exit $?)"
  .venv/bin/benchy eval --provider together --model-name "$model" \
    --tasks structured_extraction --limit 50 \
    --run-id "together_${tag}_full" --exit-policy relaxed \
    >> ".notes/bench_${tag}_struct.log" 2>&1
  echo "[$(date +%H:%M:%S)] $model structured_extraction done (exit $?)"
}

mkdir -p .notes
run_model "moonshotai/Kimi-K3" "kimik3" &
P1=$!
run_model "zai-org/GLM-5.3" "glm53" &
P2=$!
run_model "zai-org/GLM-5.2" "glm52" &
P3=$!
run_model "deepseek-ai/DeepSeek-V4-Flash-0731" "dsv4flash" &
P4=$!
run_model "MiniMaxAI/MiniMax-M3" "minimaxm3" &
P5=$!

wait $P1 $P2 $P3 $P4 $P5
echo "[$(date +%H:%M:%S)] ALL DONE"