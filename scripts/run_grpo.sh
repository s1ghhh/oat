# export CUDA_VISIBLE_DEVICES=4,5
# python examples/r1_zero_math.py --critic_type grpo --gpus 2 --vllm_gpu_ratio 0.25 --gradient-checkpointing --flash-attn --bf16 --rnd-seed --learning_rate 0.000001 --lr_warmup_ratio 0.0000001 --kl_penalty_coef 0 --num_ppo_epochs 1 --beta 0.001 --non_stop_penalty 0 --oracle_type reward --oracle countdown --pretrain Qwen/Qwen2.5-1.5B --zero-stage 3 --ref_offload --prompt_data lkevinzc/CountDownZero --train_split train --input_key input --output_key gt_answer --max-train 9999999 --num_prompt_epoch 1 --prompt_max_length 256 --sync_params_every 1 --num_samples 8 --max_step_adjustment 8 --critic_max_step_adjustment 8 --temperature 1.0 --top_p 0.9 --generate_max_length 256 --save_steps -1 --train_batch_size 128 --train_batch_size_per_device 1 --mini_train_batch_size_per_device 1 --rollout_batch_size 128 --rollout_batch_size_per_device 8 --pi_buffer_maxlen_per_device 256 --eval_batch_size 200 --eval_steps 8 --eval_temperature 0 --eval_generate_max_length 1024 --eval_split test --use-wb --wb-run-name qwen-2.5-1.5b-countdown-grpo --wb_project oat-zero --save_path ./output/oat-zero-training
# python examples/r1_zero_math.py \
#     --critic_type grpo \
#     --gpus 2 \
#     --vllm_gpu_ratio 0.4 \
#     --gradient-checkpointing \
#     --bf16 \
#     --rnd-seed \
#     --learning_rate 0.000001 \
#     --lr_warmup_ratio 0.0000001 \
#     --kl_penalty_coef 0 \
#     --num_ppo_epochs 1 \
#     --beta 0.001 \
#     --non_stop_penalty 0 \
#     --oracle_type reward \
#     --oracle countdown \
#     --pretrain Qwen/Qwen2.5-1.5B \
#     --zero-stage 3 \
#     --ref_offload \
#     --prompt_data lkevinzc/CountDownZero \
#     --train_split train \
#     --input_key input \
#     --output_key gt_answer \
#     --max-train 9999999 \
#     --num_prompt_epoch 1 \
#     --prompt_max_length 256 \
#     --sync_params_every 1 \
#     --num_samples 8 \
#     --max_step_adjustment 8 \
#     --critic_max_step_adjustment 8 \
#     --temperature 1.0 \
#     --top_p 0.9 \
#     --generate_max_length 256 \
#     --save_steps -1 \
#     --train_batch_size 16 \
#     --train_batch_size_per_device 1 \
#     --mini_train_batch_size_per_device 1 \
#     --rollout_batch_size 16 \
#     --rollout_batch_size_per_device 4 \
#     --pi_buffer_maxlen_per_device 32 \
#     --eval_batch_size 20 \
#     --eval_steps 8 \
#     --eval_temperature 0 \
#     --eval_generate_max_length 256 \
#     --eval_split test \
#     --use-wb \
#     --wb-run-name qwen-2.5-1.5b-countdown-grpo 

#     # --flash-attn \

# add this if you meet "ImportError: libpython3.9.so.1.0: cannot open shared object file: No such file or directory" and use conda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python examples/r1_zero_math.py \
    --critic_type grpo \
    --gpus 4 \
    --gpus_list 1,3,4,5 \
    --vllm_gpu_ratio 0.7 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_warmup_ratio 0.0000001 \
    --kl_penalty_coef 0 \
    --num_ppo_epochs 1 \
    --beta 0.001 \
    --non_stop_penalty 0 \
    --oracle_type reward \
    --oracle countdown \
    --pretrain Qwen/Qwen2.5-1.5B \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data lkevinzc/CountDownZero \
    --train_split train \
    --input_key input \
    --output_key gt_answer \
    --max-train 9999999 \
    --num_prompt_epoch 1 \
    --prompt_max_length 256 \
    --sync_params_every 1 \
    --num_samples 8 \
    --max_step_adjustment 8 \
    --critic_max_step_adjustment 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --generate_max_length 1024 \
    --save_steps -1 \
    --train_batch_size 128 \
    --train_batch_size_per_device 2 \
    --mini_train_batch_size_per_device 2 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 32 \
    --pi_buffer_maxlen_per_device 256 \
    --eval_batch_size 200 \
    --eval_steps 8 \
    --eval_temperature 0 \
    --eval_generate_max_length 1024 \
    --eval_split test \
    --use-wb \
    --wb-run-name qwen-2.5-1.5b-countdown-grpo 
