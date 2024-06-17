# MiLe Loss
MiLe Loss can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. We train generative language models at different scales of 468M, 1.2B, and 6.7B parameters. Experiments reveal that models incorporating the proposed MiLe Loss can gain consistent performance improvement on downstream benchmarks. Details can be found in [a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models](https://aclanthology.org/2024.findings-naacl.18.pdf)(NAACL 2024).

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
@inproceedings{su-etal-2024-mile,
    title = "{M}i{L}e Loss: a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models",
    author = "Su, Zhenpeng  and
      Lin, Zijia  and
      Baixue, Baixue  and
      Chen, Hui  and
      Hu, Songlin  and
      Zhou, Wei  and
      Ding, Guiguang  and
      W, Xing",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.18",
    pages = "250--262",
    abstract = "Generative language models are usually pre-trained on large text corpus via predicting the next token (i.e., sub-word/word/phrase) given the previous ones. Recent works have demonstrated the impressive performance of large generative language models on downstream tasks. However, existing generative language models generally neglect an inherent challenge in text corpus during training, i.e., the imbalance between frequent tokens and infrequent ones. It can lead a language model to be dominated by common and easy-to-learn tokens, thereby overlooking the infrequent and difficult-to-learn ones. To alleviate that, we propose a **MiLe Loss** function for **mi**tigating the bias of **le**arning difficulties with tokens. During training, it can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. Then it scales the training loss adaptively, trying to lead the model to focus more on the difficult-to-learn tokens. On the Pile dataset, we train generative language models at different scales of 468M, 1.2B, and 6.7B parameters. Experiments reveal that models incorporating the proposed MiLe Loss can gain consistent performance improvement on downstream benchmarks.",
}
```
