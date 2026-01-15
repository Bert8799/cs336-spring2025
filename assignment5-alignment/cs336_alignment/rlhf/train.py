"""
Train a language model on one or multiple GPUs.

To run single-GPU training:

```
python train.py
```

To run multi-GPU training, use `torchrun`. e.g., for single-node, 2 GPU:

```
export OMP_NUM_THREADS={nproc} / 2  # set number of threads per process
torchrun --standalone --nproc_per_node=2 train.py
```
"""
import os
import sys
import torch
import wandb
import random
import pathlib
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
)

from cs336_alignment.rlhf.config import RLHFConfig
from cs336_alignment.rlhf.sft_dataset import SFTDataset, iterate_batches


logger = logging.getLogger(__name__)


def _endless(loader):
    while True:
        for batch in loader:
            yield batch


def _setup_distributed():
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if not is_ddp:
        device_train = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "is_ddp": False,
            "ddp_rank": 0,
            "ddp_local_rank": 0,
            "ddp_world_size": 1,
            "device": torch.device(device_train),
            "is_master_process": True,
            "seed": 0,
        }

    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(ddp_local_rank)
    return {
        "is_ddp": True,
        "ddp_rank": ddp_rank,
        "ddp_local_rank": ddp_local_rank,
        "ddp_world_size": ddp_world_size,
        "device": torch.device("cuda", ddp_local_rank),
        "is_master_process": ddp_rank == 0,
        # Each process gets a different seed so they see different batches.
        "seed": ddp_rank,
    }


def _set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _build_loaders(
    *,
    cfg: RLHFConfig,
    tokenizer: AutoTokenizer,
    micro_batch_size: int,
    build_dev: bool,
):
    train_data = SFTDataset(
        tokenizer=tokenizer,
        dataset_path=pathlib.Path(cfg.data_dir) / cfg.sft_dataset_path / cfg.sft_train_data,
        seq_length=cfg.sft_seq_length,
        shuffle=True,
    )
    train_loader = _endless(
        iterate_batches(
            dataset=train_data,
            batch_size=micro_batch_size,
            shuffle=True,
        )
    )

    dev_loader = None
    if build_dev:
        dev_data = SFTDataset(
            tokenizer=tokenizer,
            dataset_path=pathlib.Path(cfg.data_dir) / cfg.sft_dataset_path / cfg.sft_dev_data,
            seq_length=cfg.sft_seq_length,
            shuffle=False,
        )
        dev_loader = _endless(
            iterate_batches(
                dataset=dev_data,
                batch_size=cfg.sft_batch_size,
                shuffle=False,
            )
        )

    return train_loader, dev_loader


@torch.no_grad()
def _evaluate(*, model, dev_loader, loss_fn, eval_steps: int, device: torch.device) -> float:
    model.eval()
    losses = torch.zeros(eval_steps, device=device)
    for j in range(eval_steps):
        dev_batch = next(dev_loader)
        dev_input_ids = dev_batch["input_ids"].to(device, non_blocking=True)
        dev_labels = dev_batch["labels"].to(device, non_blocking=True)
        dev_logits = model(input_ids=dev_input_ids).logits
        dev_loss = loss_fn(
            dev_logits.view(-1, dev_logits.size(-1)),
            dev_labels.view(-1),
        )
        losses[j] = dev_loss.detach()
    model.train()
    return losses.mean().item()


def _save_model(*, model, tokenizer, save_dir: str, is_ddp: bool):
    raw_model = model.module if is_ddp else model
    raw_model.save_pretrained(save_directory=save_dir)
    tokenizer.save_pretrained(save_directory=save_dir)


def sft_train(cfg: RLHFConfig):
    assert cfg.sft_batch_size % cfg.sft_gradient_accumulation_steps == 0, \
        "Batch size must be divisible by gradient accumulation steps"
    micro_batch_size = cfg.sft_batch_size // cfg.sft_gradient_accumulation_steps
    # total_steps = (len(train_data) // micro_batch_size) * cfg.sft_epochs
    total_steps = cfg.sft_train_steps
    assert total_steps % cfg.sft_eval_interval == 0, \
        "Total training steps must be divisible by evaluation interval"

    ddp = _setup_distributed()
    is_ddp = ddp["is_ddp"]
    ddp_local_rank = ddp["ddp_local_rank"]
    ddp_world_size = ddp["ddp_world_size"]
    device = ddp["device"]
    is_master_process = ddp["is_master_process"]
    seed = ddp["seed"]

    if is_master_process:
        logger.info(f"DDP training on {ddp_world_size} devices" if is_ddp else "Single device training")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    model.gradient_checkpointing_enable()
    # Required by HF when using gradient checkpointing.
    if hasattr(model, "config"):
        model.config.use_cache = False

    if is_master_process:
        logger.info(f"Model {cfg.model_name} loaded on device {device}")

    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.sft_learning_rate,
        betas=(cfg.sft_adam_beta1, cfg.sft_adam_beta2),
        eps=cfg.sft_adam_eps,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(cfg.sft_warmp_ratio * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = CrossEntropyLoss()
    
    if is_master_process:
        logger.info(
            "Total number of tokens per training step: "
            + str(
                ddp_world_size * cfg.sft_batch_size * cfg.sft_seq_length
            )
        )

    train_dataloader, dev_dataloader = _build_loaders(
        cfg=cfg,
        tokenizer=tokenizer,
        micro_batch_size=micro_batch_size,
        build_dev=is_master_process,
    )

    if is_master_process:
        logger.info("Load train data from {}".format(
            pathlib.Path(cfg.data_dir) / cfg.sft_dataset_path / cfg.sft_train_data,
        ))
        logger.info("Load dev data from {}".format(
            pathlib.Path(cfg.data_dir) / cfg.sft_dataset_path / cfg.sft_dev_data,
        ))
    _set_seed(seed)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    if is_master_process:
        logger.info("Ready to start training.")

    # NOTE: cfg.sft_epochs == 1, otherwise we would need an outer loop
    # for epoch in range(cfg.sft_epochs):

    # get first batch
    batch = next(train_dataloader)
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)

    for i in tqdm(range(total_steps), disable=not is_master_process):
        step_loss_sum = torch.tensor(0.0, device=device)
        for micro_step in range(cfg.sft_gradient_accumulation_steps):
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = (
                    micro_step == cfg.sft_gradient_accumulation_steps - 1
                )
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch = next(train_dataloader)
                next_input_ids = next_batch['input_ids'].to(device, non_blocking=True)
                next_labels = next_batch['labels'].to(device, non_blocking=True)

                micro_loss = loss_fn(
                    logits.view(-1, logits.size(-1)), # shape (batch_size * seq_length, vocab_size)
                    labels.view(-1) # shape (batch_size * seq_length)
                )
                loss = micro_loss / cfg.sft_gradient_accumulation_steps

            loss.backward()
            step_loss_sum += micro_loss.detach()
            input_ids = next_input_ids
            labels = next_labels

        if cfg.sft_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.sft_grad_clip
            )
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_float = (step_loss_sum / cfg.sft_gradient_accumulation_steps).item()
        if is_master_process:
            logger.info(f"Train step {i}, Loss: {loss_float}")
            wandb.log({
                "train/loss": loss_float,
                "train/lr": scheduler.get_last_lr()[0],
                "train_step": i + 1
            })

        if (i + 1) % cfg.sft_eval_interval == 0 and is_master_process:
            dev_loss = _evaluate(
                model=model,
                dev_loader=dev_dataloader,
                loss_fn=loss_fn,
                eval_steps=cfg.sft_eval_steps,
                device=device,
            )

            logger.info(f"Estimated validation loss: {dev_loss}")
            wandb.log({
                "eval/loss": dev_loss,
                "eval_step": i + 1
            })

            save_dir = f"{cfg.result_dir}/{cfg.sft_output_dir}/{cfg.sft_wandb_run_name}"
            _save_model(model=model, tokenizer=tokenizer, save_dir=save_dir, is_ddp=is_ddp)
            logger.info(f"Saving model weights to {save_dir}")
        
    if is_ddp:
        destroy_process_group()


if __name__ == "__main__":
    cfg = RLHFConfig()
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    # Rank 0 does logging, file creation, etc.
    is_master_process = int(os.environ["RANK"]) == 0 if is_ddp else True

    if is_master_process:
        logging.basicConfig(level=logging.INFO)
        
        # Make the directory for output if it doesn't already exist
        pathlib.Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cfg.result_dir / cfg.sft_output_dir).mkdir(parents=True, exist_ok=True)
        # Initialize wandb
        wandb.init(
            project=cfg.sft_wandb_project,
            name=cfg.sft_wandb_run_name,
            entity=cfg.sft_wandb_entity,
            config=vars(cfg),
        )

        # Setup wandb metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")
    else:
        logging.basicConfig(level=logging.ERROR)

    sft_train(cfg)