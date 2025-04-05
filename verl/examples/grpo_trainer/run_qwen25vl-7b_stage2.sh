source /data_train2/mllm/anaconda3/bin/activate mlf_verl_dev
echo "python=$(which python)"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY="{Your WANDB API KEY }" 

# login wandb
wandb login --relogin $WANDB_API_KEY

export FORMAT_REWARD_FACTOR=0.05
export LIMIT_IMAGE_NUM=1
wandb_project_name='verl_grpo'
wandb_experiment_name='7B-qwen2d5vl_stage1'
kl_coef=0.001
lr=8e-7

# Paths
# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="{Checkpoint from stage 1}"
TRAIN_FILE="{Train Data: stage 2}"
TEST_FILE="['./data/cvbench_test.parquet','./data/geoqa_test.parquet']"

output_dir=./Qwen2.5-VL-7B-grpo-verl

GPU_NUMS=`nvidia-smi -L | wc -l`
 
if [ $(($GPU_NUMS % 2)) -eq 1 ]; then
    echo "GPUs per node should be even! exist now"
    exit 1
fi

# Ray
# WORK_DIR=`dirname $(dirname $SCRIPT_DIR)` change to your path
WORK_DIR="./verl" 
echo "WORK_DIR=$WORK_DIR"
cd $WORK_DIR
RUNTIME_ENV="${WORK_DIR}/verl/trainer/runtime_env.yaml"
NNODES=$WORLD_SIZE

echo "########### run ray start ###########"
ray start --head --num-gpus $GPU_NUMS --temp-dir ~/.cache/ray --max-worker-port 12800  --runtime-env-agent-port 20100 --dashboard-agent-grpc-port 20101 --dashboard-agent-listen-port 20102 --metrics-export-port 20103
sleep 30s
ray status
echo "########### run ray end ###########"

sleep 60s
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    custom_reward_function.path=./verl/verl/utils/reward_score/custom_reward_fn.py \
    custom_reward_function.name=my_reward_fn \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$wandb_project_name \
    trainer.experiment_name=$wandb_experiment_name \
    trainer.default_local_dir=$output_dir/$wandb_experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=2
echo 'job done, now shutdown ray cluster'
ray stop --force
