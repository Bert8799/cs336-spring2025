import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .optim import AdamW, lr_cosine_scheduler, gradient_clipping
from .utils import cross_entropy_loss, get_batch, save_checkpoint, load_checkpoint


class Solver:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: np.ndarray,
        test_data: np.ndarray | None = None,
        batch_size: int = 64,
        context_length: int = 256,
        iterations: int = 20_000,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_iters: int = 1000,
        cosine_iters: int = 10_000,
        grad_clip: float | None = None,
        weight_decay: float = 0.0,
        device: torch.device = torch.device("cpu"),
        validation: bool = True,
        val_every: int = 100,
        val_iters: int = 50,
        save_every: int = 1000,
        out_path: str | None = None
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.context_length = context_length
        self.iterations = iterations
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_iters = cosine_iters
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.device = device
        self.validation = validation
        self.val_every = val_every
        self.val_iters = val_iters
        self.save_every = save_every
        self.out_path = out_path

        self.learning_rate = max_lr

        self.optim = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self._reset()

    def _reset(self):
        self.t = 0
        self.train_loss_history = []
        self.test_loss_history = []

    def _step(self):
        inputs, targets = get_batch(
            self.train_data,
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device
        )

        logits = self.model(inputs)
        loss = cross_entropy_loss(logits, targets)

        self.train_loss_history.append(loss.detach().cpu().numpy())
        self.optim.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            gradient_clipping(self.model.parameters(), self.grad_clip)

        self.optim.step()

    @torch.no_grad()
    def eval(self, is_test: bool = False):
        self.model.eval()

        total_loss = 0.0
        iter_bar = tqdm(range(self.val_iters))
        for t in iter_bar:
            inputs, targets = get_batch(
                self.test_data,
                batch_size=self.batch_size,
                context_length=self.context_length,
                device=self.device
            )
            
            logits = self.model(inputs)
            loss = cross_entropy_loss(logits, targets)
            total_loss += loss.detach().cpu().numpy()
            avg_loss = total_loss / (t + 1)
            iter_bar.set_description(f"Iter {t+1}/{self.val_iters}, Test Loss: {avg_loss:.3f}")

        if not is_test:
            self.test_loss_history.append(avg_loss)

        self.model.train()

    def train(self):
        iter_bar = tqdm(range(self.t, self.iterations))
        for t in iter_bar:
            self.learning_rate = lr_cosine_scheduler(
                it=t,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_iters=self.warmup_iters,
                cosine_iters=self.cosine_iters
            )
            for param_group in self.optim.param_groups:
                param_group['lr'] = self.learning_rate

            self._step()

            train_loss = self.train_loss_history[-1]

            if self.validation and (t + 1) % self.val_every == 0:
                self.eval()

            iter_bar.set_description(    
                f"Iter {t + 1}/{self.iterations}, Train Loss: {train_loss:.3f}"
            )

            if self.out_path is not None and (
                (t + 1) % self.save_every == 0 or t + 1 == self.iterations
            ):
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optim,
                    iteration=t + 1,
                    out=f"{self.out_path}/checkpoint_iter_{t + 1}.pt"
                )

    def load(self, path: str):
        self.t = load_checkpoint(
            model=self.model,
            optimizer=self.optim,
            checkpoint_path=path,
            device=self.device
        )

    def plot_losses(self):
        """Plots the training and test loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Train Loss')
        if self.validation:
            val_x = np.arange(self.val_every - 1, self.iterations, self.val_every)
            plt.plot(val_x, self.test_loss_history, label='Test Loss', marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Iterations')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.out_path}/loss_curve.png', dpi=300)
