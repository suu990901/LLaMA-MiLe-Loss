# Enable RDMA
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
ip=$1
rank=$2
version=seq1024
expert_name=crossentropy_468M_version${version}
mkdir -p running_log_468M
nohup accelerate launch  \
    --num_processes 16 \
    --num_machines 2 \
    --main_process_port 60199  \
    --main_process_ip $ip \
    --machine_rank $rank \
    --config_file configs/accelerate_configs/ds_stage2.yaml \
    train_lm.py \
    --train_config configs/pretrain_config_468M.yaml \
    --model_config configs/model_configs/468M.json \
    --version  $version \
> running_log_468M/${expert_name}_rank${rank}.log 2>&1 &
