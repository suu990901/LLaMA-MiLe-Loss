import time
import wandb
import torch
import logging
from torchinfo import summary
from deepspeed.ops.adam import FusedAdam
from transformers import get_cosine_schedule_with_warmup
from torch import nn
import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import math
import os
class InfoEntropyLoss(nn.Module):
    def __init__(self, gamma=1.0,reduction='mean'):
        super(InfoEntropyLoss, self).__init__()
        self.base_gamma = gamma
        self.reduction = reduction
        self.loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        self.sigma = 1

    def entropy(self,logits):
        # Detach can help the model stay stable during the training process.
        logits = logits.detach() 
        probs = F.softmax(logits,dim=-1)
        epsilon = 1e-8
        probs = torch.clamp(probs, epsilon, 1.0)
        entropy = torch.sum(-1*(probs * torch.log(probs)),dim=-1)
        return entropy

    def forward(self, inputs, targets, global_steps):
        ce_loss = self.loss_fct(inputs, targets)
        pt = self.entropy(inputs)
        gamma = self.base_gamma
        if dist.get_rank() == 0:
            wandb.log({"gamma":gamma}, commit=False)

        alpha = 1.0/(((self.sigma + pt) ** gamma).mean())
        torch.distributed.all_reduce(alpha)
        #Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        alpha = alpha / dist.get_world_size()
        loss = alpha * ((self.sigma + pt) ** gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.base_gamma = gamma
        self.reduction = reduction
        self.loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def forward(self, inputs, targets, global_steps):
        ce_loss = self.loss_fct(inputs, targets)
        pt = torch.exp(-ce_loss)
        # Detach can help the model stay stable during the training process. 
        pt = pt.detach()
        gamma = self.base_gamma
        if torch.distributed.get_rank() == 0:
            wandb.log({"gamma":gamma}, commit=False)
        
        alpha = 1 / ((1-pt)**gamma).mean() 
        torch.distributed.all_reduce(alpha)
        alpha = alpha / dist.get_world_size() 
        # Alpha is a normalization factor that allows the training loss to be comparable to the cross-entropy loss.
        loss = alpha * (1 - pt) ** gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Trainer:
    def __init__(self, config, raw_model, train_loader, validation_loader,tokenizer, accelerator):
        self.config = config
        self.raw_model = raw_model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.train_and_eval = config["train"].get("train_and_eval", False)
        self.gradient_accumulation_steps = config["train"].get(
            "gradient_accumulation_steps", 1
        )
        self.lr_scheduler_factor = (
            accelerator.num_processes / accelerator.gradient_accumulation_steps
        )
        self.log_interval = (
            self.config["log_interval"] * accelerator.gradient_accumulation_steps
        )
        self.eval_interval = (
            self.config["eval_interval"] * accelerator.gradient_accumulation_steps
        )
        self.save_interval = (
            self.config["save_interval"] * accelerator.gradient_accumulation_steps
        )
        self.work_dir = self.config["work_dir"]
        # self.get_model_info()
        if accelerator.is_main_process:
            wandb.init(project=self.config["project_name"], name=self.config.get("run_name", None))

        # get validation list
        with open(config["data"]["domain_config_path"], 'r') as f:
            domain_config = json.load(f)
        eval_domain_weights_dict = domain_config['eval_domain_weights']
        self.domain_list = list(sorted(eval_domain_weights_dict.keys()))
        if config["loss"]["use_loss"] == "crossentropy":
            self.train_loss = nn.CrossEntropyLoss(reduction='mean',ignore_index=-100)
        elif config["loss"]["use_loss"] == "focal_loss":
            gamma = config["loss"]["gamma"]
            self.train_loss = FocalLoss(gamma=gamma)
        elif config["loss"]["use_loss"] == "entropy_loss":
            gamma = config["loss"]["gamma"]
            self.train_loss = InfoEntropyLoss(gamma=gamma)

    def get_model_info(self):
        with torch.no_grad():
            summary(
                self.raw_model.cuda(),
                input_data=torch.ones(1, 64, dtype=torch.int64).cuda(),
            )

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        if self.config["train"].get("use_lora", False):
            optimizer_grouped_parameters = self.raw_model.parameters()
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config["train"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        self.optim = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.config["train"]["lr"],
            betas=(0.9, 0.95),
        )

    def get_lr_scheduler(self):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=self.config["train"]["num_warmup_steps"]
            * self.lr_scheduler_factor,
            num_training_steps=self.config["train"]["num_training_steps"]
            * self.lr_scheduler_factor,
        )
       

    def prepare(self):
        (
            _,
            self.model,
            self.optim,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_loader, self.raw_model, self.optim, self.scheduler
        )
        self.optim.zero_grad()
        self.global_step = 0
        try:
            global_step_path = self.work_dir + os.sep + "global_step"
            with open(global_step_path,"r") as f:
                self.global_step = f.readline()
                self.global_step = int(self.global_step)
            global_step = str(self.global_step)
            save_path = f"{self.work_dir}/ckpt{global_step}"
            self.accelerator.load_state(save_path)
            logging.warning("Restored ckpt from {}".format(self.work_dir))
        except:
            logging.warning("No ckpt found in {}".format(self.work_dir))
        if self.global_step > 0:
            skip_steps = self.global_step * self.gradient_accumulation_steps
            logging.warning("Skiped {} steps.".format(skip_steps))
            self.train_loader_skiped = self.accelerator.skip_first_batches(
                self.train_loader, num_batches=skip_steps
            )
            self.data_step = skip_steps
        else:
            self.train_loader_skiped = self.train_loader
        
        self.accelerator.wait_for_everyone()

    def train_step(self, batch):
        labels = batch.pop("labels")
        # batch.pop("attention_mask")
        out = self.model(**batch)
        logits = out.logits

        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = self.train_loss
        _, _, vocab_size = shift_logits.size()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if self.config["loss"]["use_loss"]== "crossentropy":
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = loss_fct(shift_logits, shift_labels,self.data_step)
        losses = {"total_loss": loss}
        self.accelerator.backward(loss)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()
        return losses

    def train(self):
        self.data_step = 0
        self.get_optimizer()
        self.get_lr_scheduler()
        self.prepare()
        self.start_time = time.time()
        self.epoch = 0
        # self.eval()
        while True:
            if self.data_step >= self.config["train"]["num_training_steps"]:
                break
            if self.epoch == 0:
                train_loader = self.train_loader_skiped
            else:
                train_loader = self.train_loader
            for batch in train_loader:
                # end training
                if self.data_step >= self.config["train"]["num_training_steps"]:
                    global_steps = str(self.global_step)
                    save_path = f"{self.work_dir}/ckpt{global_steps}"
                    self.accelerator.save_state(save_path)
                    global_step_path = self.work_dir + os.sep + "global_step"
                    with open(global_step_path,"w") as f:
                        f.writelines(global_steps)  
                    torch.distributed.barrier()
                    time.sleep(300)
                if self.data_step >= self.config["train"]["num_training_steps"]:
                    break
                # data to device
                for k, v in batch.items():
                    if isinstance(v, list):
                        v = torch.stack(v)
                    batch[k] = v.to(self.accelerator.device, non_blocking=True)
                self.model.train()
                # train step
                with self.accelerator.accumulate(self.model):
                    losses = self.train_step(batch)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                # log
                if (
                    self.data_step % self.log_interval == 0
                    and self.data_step > 0
                    and self.accelerator.is_main_process
                ):
                    self.log(losses)
                # eval/vis model output
                if (
                    (self.data_step+1) % self.eval_interval == 0
                    and self.train_and_eval
                ):
                    self.eval()
                torch.distributed.barrier()
                # save state
                if (self.data_step+1) % self.save_interval == 0 and self.data_step > 0:
                    global_steps = str(self.global_step)
                    save_path = f"{self.work_dir}/ckpt{global_steps}"
                    self.accelerator.save_state(save_path)
                    global_step_path = self.work_dir + os.sep + "global_step"
                    with open(global_step_path,"w") as f:
                        f.writelines(global_steps)
                self.data_step += 1
            self.epoch += 1
        wandb.finish()

    def log(self, losses):
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = (
            self.config["train"]["train_batch_size"]
            * self.log_interval
            * self.config["data"]["seq_length"]
        )
        wandb.log({"Training/Token per second per gpu": tokens / cost_time},commit=False)
        for k, v in losses.items():
            wandb.log({"Losses/{}".format(k): v},commit=False)
        current_lr = self.optim.param_groups[0]["lr"]
        wandb.log({"Training/LR": current_lr},commit=False)
        if self.optim.scaler is not None:
            wandb.log({"Training/Loss Scale": self.optim.scaler.get_scale()},commit=False)
        wandb.log({"Training/Data Step": self.data_step},commit=False)
        wandb.log({"Training/Global Step": self.global_step},commit=False)
        wandb.log({"Training/Epoch": self.epoch})
        self.accelerator.print(
            "Epoch: {}, Global Step: {}, Data Step: {}, Loss: {}, Token per second per gpu: {}".format(
                self.epoch,
                self.global_step,
                self.data_step,
                losses["total_loss"],
                tokens / cost_time,
            )
        )

    @torch.no_grad()
    def eval(self):
        # pass
        # validation_loader = self.accelerator.prepare(self.validation_loader)
        validation_loader = self.validation_loader
        model = self.model
        model.eval()
  
        loss_fn = nn.CrossEntropyLoss(reduction="sum",ignore_index=-100)
        losses = torch.zeros(len(self.domain_list)).cuda()
        tokencounts = torch.zeros(len(self.domain_list)).cuda()
        examplecounts = torch.zeros(len(self.domain_list)).cuda()

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(validation_loader)):
            # data to device
            for k, v in inputs.items():
                if isinstance(v, list):
                    v = torch.stack(v)
                inputs[k] = v.to(self.accelerator.device, non_blocking=True)
            # Prediction step
            domain_ids = inputs.pop("domain_id")
            domain_ids = domain_ids.cpu()
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits

            if isinstance(logits, tuple):
                logits = logits[0]

            # compute losses per domain
            for domain_idx, domain_name in enumerate(self.domain_list):
                domain_mask = (domain_ids == domain_idx)
                examplecounts[domain_idx] = examplecounts[domain_idx] + domain_mask.sum()
                
                if domain_mask.sum() > 0:
                    domain_labels = labels[domain_mask]
                    domain_preds = logits[domain_mask]
                    domain_labels = domain_labels[:, 1:].contiguous().view(-1)
                    domain_preds = domain_preds[:, :-1, :].contiguous().view(-1, domain_preds.size(-1))
                    losses[domain_idx] = losses[domain_idx] + loss_fn(domain_preds, domain_labels)
                    tokencounts[domain_idx] = tokencounts[domain_idx] + (domain_labels != -100).sum()

        torch.distributed.all_reduce(losses)
        torch.distributed.all_reduce(tokencounts)
        torch.distributed.all_reduce(examplecounts)

        # losses/preds/labels on CPU (final containers)
        per_domain_losses = {domain_name: losses[domain_idx].item()
                             for domain_idx, domain_name in enumerate(self.domain_list) if tokencounts[domain_idx] > 0} 
        per_domain_tokencounts = {domain_name: tokencounts[domain_idx].item()
                                  for domain_idx, domain_name in enumerate(self.domain_list) if tokencounts[domain_idx] > 0} 
        per_domain_examplecounts = {domain_name: examplecounts[domain_idx].item()
                                    for domain_idx, domain_name in enumerate(self.domain_list) if tokencounts[domain_idx] > 0} 
         
        # normalize
        per_domain_losses = {domain_name: per_domain_losses[domain_name] / per_domain_tokencounts[domain_name]
                             for domain_name in per_domain_losses.keys()}

        metrics = {f"{domain_name}_log_ppl": per_domain_losses[domain_name]
                   for domain_name in per_domain_losses.keys()}

        metrics["uniform_avg_log_ppl"] = np.mean(list(per_domain_losses.values()))
        metrics["worst_case_log_ppl"] = np.amax(list(per_domain_losses.values()))

        output = {**metrics, **{"val_step": self.global_step}}
        if self.accelerator.is_main_process:
            wandb.log(output)
        self.accelerator.print(
            str(output)
        )
