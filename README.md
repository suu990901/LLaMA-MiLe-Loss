# InfoEntropy Loss
InfoEntropy Loss can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. Details can be found in [InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models](https://arxiv.org/abs/2310.19531). We train generative language models at different scales of 468M, 1.2B, and 6.7B parameters. Experiments reveal that models incorporating the proposed InfoEntropy Loss can gain consistent performance improvement on downstream benchmarks.

## Dependencies

Dependencies can be installed by running the codes below. 
```
pip install -r requirements.txt
```
We use flash_attn==2.0.4 to accelerate training. flash_attn needs to be compiled according to the environment; please refer to [flash-attention](https://github.com/Dao-AILab/flash-attention) to install a version suitable for your system.

## Pre-training

### Data Preparation

[The Pile](https://arxiv.org/abs/2101.00027) are used as unsupervised pre-training corpus. You can download the data from [EleutherAI](https://pile.eleuther.ai). Then, run the following example script to perform data preprocessing on The Pile, which divides the Pile data into multiple domains and tokenizes it:

```
bash make_data.sh
bash merge_split_data.sh
```

### Pre-training

We train a 436M model using 16 GPUs, a 1.2B model using 32 GPUs, and a 6.7B model using 128 GPUs.
```
bash run_entropy_468M.sh 
bash run_entropy_1B.sh
bash run_entropy_7B.sh
```
Among them, run_entropy_468M.sh, run_entropy_1B.sh, and run_entropy_7B.sh, correspond to the startup scripts for the 468M, 1.2B, and 6.7B models, respectively. Specifically, the contents of run_entropy_7B.sh are as follows.
```
ip=$1
rank=$2
gamma=1.0
version=seq2048
expert_name=entropy_7B_version${version}_gamma${gamma}
mkdir -p running_log_7B
nohup accelerate launch  \
    --num_processes 128 \
    --num_machines 16 \
    --main_process_port 60199  \
    --main_process_ip $ip \
    --machine_rank $rank \
    --config_file configs/accelerate_configs/ds_stage2.yaml \
    train_lm.py \
    --train_config configs/pretrain_config_entropy_7B.yaml \
    --model_config configs/model_configs/7B.json \
    --gamma $gamma \
    --version  $version \
> running_log_7B/${expert_name}_rank${rank}.log 2>&1 &

```
In some cases, you may need to specify the following parameters for multi-node parallelism:
```
--main_process_ip
--main_process_port
--num_processes
--num_machines
--machine_rank
```

If you want to use [wandb](https://wandb.ai/) to visualize the training process, you can use:
```
wandb login
wandb online
```

### Evaluation

We use toolkit [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate zero-shot and few-shot performance. Since we are using DeepSpeed, for evaluation, we need to run zero_to_fp32.py to extract fp32 consolidated weights from a DeepSpeed checkpoint. zero_to_fp32.py is saved automatically by Deepspeed, and you will find it in the model's save directory.
```
python zero_to_fp32.py ${deepspeed_ckpt_path} ${model_path}/pytorch_model.bin
cp configs/model_configs/7B.json ${model_path}/config.json
```
Then you can evaluate the model as follows:
```
cd lm-evaluation-harness
python main.py --model hf-causal --model_args pretrained=${model_path} --tasks ${tasks} --batch_size {batch_size} --num_fewshot ${shot} --device cuda:${gpu} 
```
For more evaluation details, please refer to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Citation
If you find our work useful, please consider to cite our paper.
```
@article{su2023infoentropy,
  title={InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models},
  author={Su, Zhenpeng and Wu, Xing and Bai, Xue and Lin, Zijia and Chen, Hui and Ding, Guiguang and Zhou, Wei and Hu, Songlin},
  journal={arXiv preprint arXiv:2310.19531},
  year={2023}
}
```
