# Check for docker availability at 
sudo docker run -it -d -p 8080:8080 zenmldocker/zenml-server --name zenml

# Run eval for 1 model
(lm-evaluation-harness) mauro@clustercito:~/dev/lm-evaluation-harness$ lm_eval --model hf     --model_args pretrained=google/gemma-3n-E4B,dtype=float16,max_length=8192     --tasks latam     --device cuda     --batch_size auto:4     --output_path /home/mauro/dev/lm-evaluation-harness/output     --log_samples --wandb_args entity=surus-lat,project=LATAM-leaderboard

# upload data to database
~/dev/leaderboard$ uv run run_pipeline.py

for both calls, uses .env variables 

