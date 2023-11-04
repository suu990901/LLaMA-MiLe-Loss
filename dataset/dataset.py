import os
import json
import math
import torch
import torch.distributed as dist
import random
import numpy as np
from glob import glob
from pathlib import Path
from datasets import (
    load_dataset, 
    load_from_disk, 
    Dataset, 
    IterableDataset, 
    interleave_datasets, 
    concatenate_datasets
)
from functools import partial
import logging
logger = logging.getLogger(__name__)
if logger.level != logging.INFO:
    logger.setLevel(logging.INFO)

# Define `n_shards` of iterable datasets

def split_sequence_gen(seq_length):
    def split_sequence(batch):
        input_ids = batch["input_ids"][0]
        out = []
        while len(input_ids) >= (1 + len(out)) * seq_length:
            out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
        return {"input_ids": out}

    return split_sequence

def concat_multiple_sequence_gen(seq_length, pad_token_id):
    def _concat_sequence(feature):
        concat_input_ids = torch.cat(feature, dim=0)
        length = concat_input_ids.shape[0]
        chunks = math.ceil(length / seq_length)
        pad_length = chunks * seq_length - length
        pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_token_id
        concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)
        input_ids = torch.chunk(concat_input_ids, chunks)
        return input_ids

    def concat_multiple_sequence(batch):
        return {k: _concat_sequence(v) for k, v in batch.items()}

    return concat_multiple_sequence

def get_labels_gen(pad_token_id):
    def get_labels(line):
        input_ids = line["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        return {"labels": labels}
    return get_labels


def get_domain_datasets(
        preprocessed_dir,
        split='train',
    ):
    preprocessed_dir = Path(preprocessed_dir) / split
    domain_dirs = list(sorted(preprocessed_dir.iterdir()))  # Fix: Keep loading order of each shard
    
    # streaming = True if split == "train" else False
    streaming = True
    domain_ds = {}
    world_size = dist.get_world_size()
    for domain_idx, domain_dir in enumerate(domain_dirs):
        if dist.get_rank() in [-1, 0]:
            logger.info(f"Loading [{domain_idx+1}/{len(domain_dirs)}] domain {domain_dir.name} from {str(domain_dir)}")
        
        shard_files = list(sorted([i for i in domain_dir.iterdir() if i.is_file()]))
        shard_files = [ str(i) for i in shard_files ]

        if world_size is not None and split=="train":
            num_shards = len(shard_files)
            shard_files = shard_files[: num_shards // world_size * world_size]

        shard_files = {split:shard_files}
        domain_ds[domain_dir.name] = load_dataset("json",data_files=shard_files,split=split,streaming=streaming)

    return domain_ds

def _average_weights(weights):
    if isinstance(weights, list):
        _sum = sum(weights)
        return [i/_sum for i in weights]
    elif isinstance(weights, dict):
        _sum = sum(weights.values())
        return {k: v/_sum for k, v in weights.items()}
    elif isinstance(weights, np.ndarray):
        _sum = sum(weights)
        return weights / _sum
    else:
        raise NotImplementedError()

def construct_train_dataset(
        dataset_config: dict,
        tokenizer,
    ):
    # Loading datasets of each domains
    domain_ds = get_domain_datasets(
                    preprocessed_dir=dataset_config["dataset_dir"],
                    split='train',
                )

    # Load domain weights from local file
    with open(dataset_config["domain_config_path"], 'r') as f:
        domain_config = json.load(f)
        train_domain_weights_dict = _average_weights(domain_config['train_domain_weights'])

    # whenever we convert dict to array, we sort by key
    domain_list = list(sorted(train_domain_weights_dict.keys()))
    num_domains = len(domain_list)

    # data script could change tokenizer shape
    seed = dataset_config["seed"] + dist.get_rank()
    random.seed(seed)
    full_dataset: IterableDataset = interleave_datasets(
                    datasets=[domain_ds[_domain] for _domain in domain_list],
                    probabilities=[train_domain_weights_dict[_domain] for _domain in domain_list],
                    seed=seed,
                    stopping_strategy='all_exhausted'
                )
    # For doremi datasets
    if 'domain_id' in full_dataset.column_names:
        full_dataset = full_dataset.remove_columns('domain_id')
    
    # Convert to torch.tensor
    full_dataset = full_dataset.map(lambda line: {k: torch.tensor(v) for k, v in line.items()})
    
    # concat multiple sequence，1024 -> 2048
    if dataset_config["concat_multiple_sequence"]:
        full_dataset = full_dataset.map(
            concat_multiple_sequence_gen(dataset_config["seq_length"], tokenizer.pad_token_id),
            batched=True,
            batch_size=dataset_config["num_sequences"],
            drop_last_batch=True,
        )

    # add label
    full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_token_id))
    
    return full_dataset


def add_domain_id_fn(example, domain_idx):
    example['domain_id'] = domain_idx
    return example


def construct_eval_dataset(
        dataset_config: dict,
        tokenizer,
    ):
    # Loading datasets of each domains
    domain_ds = get_domain_datasets(
                    preprocessed_dir=dataset_config["dataset_dir"],
                    split='validation',
                )

    # Load domain weights from local file
    with open(dataset_config["domain_config_path"], 'r') as f:
        domain_config = json.load(f)
        eval_domain_weights_dict = _average_weights(domain_config['eval_domain_weights'])

    # whenever we convert dict to array, we sort by key
    domain_names = list(sorted(eval_domain_weights_dict.keys()))
    domain_to_idx = {domain_names[i]: i for i in range(len(domain_names))}

    # add domain ids
    domain_ds_ls = []
    for domain_name in domain_names:
        domain_idx = domain_to_idx[domain_name]
        one_domain_ds = domain_ds[domain_name]

        # add domain_id if necessary
        # if add_domain_id:
        if True:
            one_domain_ds = one_domain_ds.map(partial(add_domain_id_fn, domain_idx=domain_idx))
        domain_ds_ls.append(one_domain_ds)

    # instead of interleaving, run through each dataset
    def data_generator(shards):
        for shard in shards:
            for ex in shard:
                yield ex

    mixed_ds_shard = IterableDataset.from_generator(data_generator, gen_kwargs={'shards': domain_ds_ls})
    mixed_ds_shard = mixed_ds_shard.map(lambda line: {k: torch.tensor(v) for k, v in line.items()})
    mixed_ds_shard = mixed_ds_shard.map(get_labels_gen(tokenizer.pad_token_id))
    return mixed_ds_shard