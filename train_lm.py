import yaml
import math
import logging
from absl import app
from absl import flags
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import set_seed
from dataset.dataset import construct_train_dataset,construct_eval_dataset
from utils.trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config", None, "Training config path")
flags.DEFINE_string(
    "model_config", "configs/model_configs/7B.json", "Model config path"
)
flags.DEFINE_string(
    "gamma", None, "gamma for Loss"
)
flags.DEFINE_string(
    "version", None, "running version"
)
def main(argv):
    with open(FLAGS.train_config, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    if FLAGS.gamma !=None:
        config["loss"]["gamma"] = float(FLAGS.gamma)
    config["work_dir"] = config["work_dir"] + f"gamma_{FLAGS.gamma}"
    config["run_name"] = config["run_name"] + f"gamma_{FLAGS.gamma}"
    if  FLAGS.version is not None:
        config["work_dir"] = config["work_dir"] + f"version_{FLAGS.version}"
        config["run_name"] = config["run_name"] + f"version_{FLAGS.version}"

    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get(
            "gradient_accumulation_steps", 1
        )
    )
    data_config = config["data"]
    seed = data_config["seed"] + dist.get_rank()
    set_seed(seed = seed)
    tokenizer = AutoTokenizer.from_pretrained(data_config["tokenizer_model_path"], use_fast=False)
    tokenizer.pad_token = tokenizer.bos_token   

    # build train dataloader
    train_dataset = construct_train_dataset(data_config, tokenizer)
    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        num_workers=config["train"]["train_num_workers"],
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )
    # build valitdation dataloader
    validation_dataset = construct_eval_dataset(data_config, tokenizer)
    validation_dataset = split_dataset_by_node(
        validation_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["train"]["train_batch_size"]*2,
        num_workers=1,
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )

    model_config = AutoConfig.from_pretrained(FLAGS.model_config)
    vocab_size = tokenizer.vocab_size
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = None 
    if config["train"]["ckpt"] is not None:
        raw_model = AutoModelForCausalLM.from_pretrained(
            config["train"]["ckpt"], config=model_config
        )
        logging.warning("Loaded ckpt from: {}".format(config["train"]["ckpt"]))
    else:
        raw_model = AutoModelForCausalLM.from_config(model_config)
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()
    trainer = Trainer(config, raw_model, train_loader, validation_loader,tokenizer, accelerator)
    trainer.train()

if __name__ == "__main__":
    app.run(main)
