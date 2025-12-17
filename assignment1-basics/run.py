import torch
import numpy as np
from cs336_basics.train_bpe import hf_train_bpe
from cs336_basics.tokenizer import HuggingFaceTokenizer
from cs336_basics.solver import Solver
from cs336_basics.transformer import TransformerLM
from cs336_basics.parser import parse_args


def run_train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    merge_outpath=None,
    vocab_outpath=None,
    model_outpath=None,
    type: str = "custom" or "hf" or None,
) -> tuple[dict[int, str], list[tuple[bytes, bytes]]]:
    if type is None:
        raise ValueError("Type must be specified as 'custom' or 'hf'.")
    elif type == "hf":
        return hf_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            merge_outpath=merge_outpath,
            vocab_outpath=vocab_outpath,
            model_outpath=model_outpath
        )


def run_tokenize_bpe(
    input_path: str,
    vocab_filepath: str | None = None,
    merge_filepath: str | None = None,
    model_filepath: str | None = None,
    type: str = "custom" or "hf" or None,
):
    if type is None:
        raise ValueError("Type must be specified as 'custom' or 'hf'.")
    elif type == "hf":
        tokenizer = HuggingFaceTokenizer(model_filepath=model_filepath)
        token_ids = tokenizer.encode_lines(input_path)

        np.array(token_ids, dtype=np.uint16).tofile(f"{input_path}.bin")


def run_train_lm():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        context_length=args.context_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # train_data = np.memmap(f'{args.data_path}/train.bin', dtype=np.uint16, mode='r')
    test_data = np.memmap(f'{args.data_path}/test.bin', dtype=np.uint16, mode='r')
    print("Data loaded.")

    solver = Solver(
        model=model,
        train_data=None,
        test_data=test_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        iterations=args.iterations,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_iters=args.cosine_iters,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        device=device,
        validation=args.validation,
        val_every=args.val_every,
        val_iters=args.val_iters,
        save_every=args.save_every,
        out_path=args.out_path
    )

    solver.load(path=f"{args.out_path}/checkpoint_iter_20000.pt")
    print("Checkpoint loaded.")

    # print("Starting training.")
    # solver.train()
    # print("Training completed.")

    solver.eval()
    print("Evaluation completed.") # 1.440


def run_inference():    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        context_length=args.context_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        device=device
    )

    solver = Solver(
        model=model,
        train_data=None,
        test_data=None,
        device=device
    )

    solver.load(path=f"{args.out_path}/checkpoint_iter_20000.pt")

    tokenizer = HuggingFaceTokenizer(model_filepath=args.tokenizer_model_path)
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_length = 50
    generated_ids = solver.inference(
        input_ids=input_tensor,
        output_length=output_length,
        p=0.9,
        t=1.0
    )

    generated_ids = generated_ids[0].cpu().numpy().tolist()
    generated_text = tokenizer.decode(generated_ids)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__": 
    run_inference()

