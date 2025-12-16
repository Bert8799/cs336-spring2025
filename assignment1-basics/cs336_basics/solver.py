import torch
from tqdm import tqdm
import numpy as np
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
        num_epochs: int = 2,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_iters: int = 1000,
        cosine_iters: int = 10000,
        grad_clip: float | None = None,
        weight_decay: float = 0.0,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        print_every: int = 100,
        out: str = './results'
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.context_length = context_length
        self.num_epochs = num_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_iters = cosine_iters
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.print_every = print_every
        self.out = out

        self.learning_rate = max_lr

        self.optim = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self._reset()

    def _reset(self):
        self.epoch = 0
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
    def eval(self):
        self.model.eval()
        num_test = self.test_data.shape[0]
        num_iterations = max(num_test // self.batch_size, 1)
        total_loss = 0.0

        for _ in range(num_iterations):
            inputs, targets = get_batch(
                self.test_data,
                batch_size=self.batch_size,
                context_length=self.context_length,
                device=self.device
            )
            
            logits = self.model(inputs)
            loss = cross_entropy_loss(logits, targets)
            total_loss += loss.item()

        avg_loss = total_loss / num_iterations
        self.test_loss_history.append(avg_loss)

        self.model.train()
        return avg_loss

    def train(self):
        num_train = self.data.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
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

            if self.verbose and (t + 1) % self.print_every == 0:
                avg_loss = np.mean(self.train_loss_history[-self.print_every:])
                print(f"Iteration {t + 1}/{num_iterations}, Train Loss: {avg_loss:.4f}", end='')

                if self.test_data is not None:
                    test_loss = self.eval()
                    print(f", Test Loss: {test_loss:.4f}")
                else:
                    print()
            
            if (t + 1) % (10 * self.print_every) == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optim,
                    iteration=t + 1,
                    out=f'{self.out}/checkpoint_iter_{t + 1}.pt'
                )

        print(f"\nTrain Loss: {self.train_loss_history[-1]:.4f}", end='')
        if self.test_data is not None:
            test_loss = self.eval()
            print(f", Test Loss: {test_loss:.4f}")
        else:
            print()

        torch.save({'model_state_dict': self.model.state_dict(),}, f'{self.out}/final_model.pt')

    def load(self, path: str):
        load_checkpoint(
            model=self.model,
            optimizer=self.optim,
            path=path,
            device=self.device
        )

