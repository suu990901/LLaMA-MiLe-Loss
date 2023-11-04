# Enable RDMA
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
ip=$1
rank=$2
gamma=$3
version=seq1024
expert_name=focal_1B_version${version}_gamma${gamma}
mkdir -p running_log_1B
nohup accelerate launch  \
    --num_processes 32 \
    --num_machines 4 \
    --main_process_port 60199  \
    --main_process_ip $ip \
    --machine_rank $rank \
    --config_file configs/accelerate_configs/ds_stage2.yaml \
    train_lm.py \
    --train_config configs/pretrain_config_focal_1B.yaml \
    --model_config configs/model_configs/1B.json \
    --gamma $gamma \
    --version  $version \
> running_log_1B/${expert_name}_rank${rank}.log 2>&1 &
