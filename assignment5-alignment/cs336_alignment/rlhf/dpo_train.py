import os
import torch
import wandb
import logging
from tqdm import tqdm
from torch.amp import autocast
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Local imports
from cs336_alignment.rlhf.config import RLHFConfig
from cs336_alignment.rlhf.sft_train import _set_seed
from cs336_alignment.rlhf.dpo_dataset import DPODataset, DPODataCollator

# Import from our new utils
from cs336_alignment.rlhf.dpo_utils import (
    get_response_log_probs, 
    dpo_loss, 
    to_device, 
    create_dpo_model_pair, 
    iterate_batches
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DPOTrainer:
    def __init__(self, cfg: RLHFConfig):
        self.cfg = cfg
        self.policy_device = torch.device(cfg.dpo_device_train)
        self.ref_device = torch.device(cfg.dpo_device_ref)
        self.micro_batch_size = cfg.dpo_batch_size // cfg.dpo_gradient_accumulation_steps
        
        # 1. Models (Delegated to Factory)
        self.policy, self.ref_model = create_dpo_model_pair(
            cfg.dpo_model_name, self.policy_device, self.ref_device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.dpo_model_name)
        
        # 2. Optimization
        self.optimizer = RMSprop(self.policy.parameters(), lr=cfg.dpo_learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.dpo_warmup_steps,
            num_training_steps=cfg.dpo_train_steps
        )
        self.data_collator = DPODataCollator(pad_token_id=self.tokenizer.pad_token_id)

    def train(self):
        logger.info("⚡️ Preparing DataLoaders")
        train_ds = DPODataset(
            f"{self.cfg.data_dir}/{self.cfg.dpo_dataset_path}/{self.cfg.dpo_train_data}", 
            self.tokenizer, shuffle=True
        )
        eval_ds = DPODataset(
            f"{self.cfg.data_dir}/{self.cfg.dpo_dataset_path}/{self.cfg.dpo_eval_data}", 
            self.tokenizer, shuffle=False
        )

        train_iter = iterate_batches(
            train_ds, self.micro_batch_size, self.data_collator, 
            shuffle=True, drop_last=True, num_workers=4
        )

        self.policy.train()
        logger.info(f"🔥 Starting DPO Training for {self.cfg.dpo_train_steps} steps")

        # Prefetch first batch
        batch = next(train_iter)
        
        # Pipeline: Move ref data early
        ref_chosen = to_device(batch, self.ref_device, "chosen")
        ref_rejected = to_device(batch, self.ref_device, "rejected")

        data_bar = tqdm(range(self.cfg.dpo_train_steps))
        for step in data_bar:
            metrics = {"loss": 0.0, "margin": 0.0, "acc": 0.0}
            
            for _ in range(self.cfg.dpo_gradient_accumulation_steps):
                # --- Step 1: Reference Model (Ref Device) ---
                with torch.no_grad():
                    with autocast(device_type=self.ref_device.type, dtype=torch.bfloat16):
                        ref_logr_chosen = get_response_log_probs(self.ref_model, **ref_chosen)
                        ref_logr_rejected = get_response_log_probs(self.ref_model, **ref_rejected)
                
                # --- Step 2: Policy Model (Policy Device) ---
                # Move data to policy device just in time
                policy_chosen = to_device(ref_chosen, self.policy_device)
                policy_rejected = to_device(ref_rejected, self.policy_device)
                
                # Ensure ref logs are on policy device for loss calc
                ref_logr_chosen = ref_logr_chosen.to(self.policy_device)
                ref_logr_rejected = ref_logr_rejected.to(self.policy_device)

                with autocast(device_type=self.policy_device.type, dtype=torch.bfloat16):
                    pi_logr_chosen = get_response_log_probs(self.policy, **policy_chosen)
                    pi_logr_rejected = get_response_log_probs(self.policy, **policy_rejected)
                    
                    # Pre-fetch next batch while GPU is busy computing
                    batch = next(train_iter)
                    ref_chosen = to_device(batch, self.ref_device, "chosen")
                    ref_rejected = to_device(batch, self.ref_device, "rejected")
                    
                    # Compute Loss (Pure Math)
                    loss, margin, acc = dpo_loss(
                        (pi_logr_chosen, pi_logr_rejected),
                        (ref_logr_chosen, ref_logr_rejected),
                        beta=self.cfg.dpo_beta
                    )
                    
                    loss_norm = loss / self.cfg.dpo_gradient_accumulation_steps
                
                loss_norm.backward()
                
                metrics["loss"] += loss_norm.item() * self.cfg.dpo_gradient_accumulation_steps
                metrics["margin"] += margin.item()
                metrics["acc"] += acc.item()

            # --- Optimization ---
            metrics = {k: v / self.cfg.dpo_gradient_accumulation_steps for k, v in metrics.items()}
            
            if self.cfg.dpo_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.dpo_grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            data_bar.set_description("Step: [{}/{}] Loss: {:.3f} Acc: {:.3f}%"
                                     .format(step + 1, self.cfg.dpo_train_steps,
                                             metrics["loss"],
                                             metrics["acc"] * 100))

            wandb.log({
                "train/loss": metrics["loss"],
                "train/reward_margin": metrics["margin"],
                "train/accuracy": metrics["acc"],
                "train/lr": self.scheduler.get_last_lr()[0],
                "train_step": step + 1
            })

            # --- Evaluation ---
            if (step + 1) % self.cfg.dpo_eval_interval == 0:
                self.evaluate(eval_ds, step + 1)
                self.policy.train()
            
            # --- Checkpointing ---
            if (step + 1) % self.cfg.dpo_save_interval == 0:
                self.save_checkpoint(step + 1)

    def evaluate(self, dataset, step):
        logger.info(f"Running Eval at step {step}")
        self.policy.eval()
        
        eval_loader = DataLoader(
            dataset, batch_size=self.micro_batch_size, 
            collate_fn=self.data_collator, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        eval_iter = iter(eval_loader)
        
        metrics_sum = {"loss": 0.0, "margin": 0.0, "acc": 0.0}
        count = 0
        
        # Prefetch first batch
        batch = next(eval_iter)
        ref_chosen = to_device(batch, self.ref_device, "chosen")
        ref_rejected = to_device(batch, self.ref_device, "rejected")
        
        with torch.no_grad():
            for _ in range(self.cfg.dpo_eval_steps):
                # Ref Forward
                with autocast(device_type=self.ref_device.type, dtype=torch.bfloat16):
                    ref_logr_chosen = get_response_log_probs(self.ref_model, **ref_chosen)
                    ref_logr_rejected = get_response_log_probs(self.ref_model, **ref_rejected)

                # Policy Forward
                # Move data to policy device
                policy_chosen = to_device(ref_chosen, self.policy_device)
                policy_rejected = to_device(ref_rejected, self.policy_device)
                
                # Align devices for loss
                ref_logr_chosen = ref_logr_chosen.to(self.policy_device)
                ref_logr_rejected = ref_logr_rejected.to(self.policy_device)
                
                with autocast(device_type=self.policy_device.type, dtype=torch.bfloat16):
                    pi_logr_chosen = get_response_log_probs(self.policy, **policy_chosen)
                    pi_logr_rejected = get_response_log_probs(self.policy, **policy_rejected)
                    
                    # Pre-fetch next batch
                    batch = next(eval_iter)
                    ref_chosen = to_device(batch, self.ref_device, "chosen")
                    ref_rejected = to_device(batch, self.ref_device, "rejected")

                    loss, margin, acc = dpo_loss(
                        (pi_logr_chosen, pi_logr_rejected),
                        (ref_logr_chosen, ref_logr_rejected),
                        beta=self.cfg.dpo_beta
                    )
                
                metrics_sum["loss"] += loss.item()
                metrics_sum["margin"] += margin.item()
                metrics_sum["acc"] += acc.item()
                count += 1

        avg = {k: v / count for k, v in metrics_sum.items()}
        logger.info(f"Eval Result: Loss: {avg['loss']:.3f}, Margin: {avg['margin']:.3f}, Acc: {avg['acc']:.3f}")
        wandb.log({f"eval/{k}": v for k, v in avg.items()} | {"eval_step": step})

    def save_checkpoint(self, step):
        save_dir = f"{self.cfg.result_dir}/{self.cfg.dpo_output_dir}/step_{step}"
        logger.info(f"Saving checkpoint to {save_dir}")
        self.policy.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    config = RLHFConfig()
    _set_seed(config.seed)
    
    os.makedirs(f"{config.result_dir}/{config.dpo_output_dir}", exist_ok=True)
    
    wandb.init(
        project=config.dpo_wandb_project,
        name=config.dpo_wandb_run_name,
        config=vars(config)
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    trainer = DPOTrainer(config)
    trainer.train()