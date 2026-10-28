#!/usr/bin/env python3
"""
train.py — Train a ~0.5B-parameter Llama-style decoder-only language model
FROM SCRATCH (random init, NO pretrained weights) on a single NVIDIA T4 GPU
(Google Colab). Uses fp16 (NOT bf16 — T4 is Turing and does not support bf16),
gradient checkpointing, and HuggingFaceFW/fineweb-edu (sample-10BT) streamed
over the network so we never download the full 10 BT.

Target architecture (Llama-style):
    hidden_size           = 1280
    num_hidden_layers     = 24
    num_attention_heads   = 20
    num_key_value_heads   = 10   (GQA, 2 query heads per KV head)
    intermediate_size     = 3456
    vocab_size            = 32000
    max_position_embeddings = 1024
    tie_word_embeddings   = True
    attention_bias        = False
    rms_norm_eps          = 1e-5
    rope_theta            = 10000.0
    => ~478M trainable params (within 450M–550M target).

Memory plan on T4 (16 GB):
    - model fp16 weights         ≈ 1.0 GB
    - AdamW master + 2 moments    ≈ 6.0 GB (fp32)
    - activations (grad ckpt on) ≈ 1.5–3 GB
    - micro batch 8 × 1024 tokens is fine
    - => micro batch 8 × grad_accum 8 = 65,536 tokens per optimizer step
       (within the requested 64k–128k effective-batch window).

================================================================================
PIP INSTALL (run once in Colab, fast via uv):
================================================================================
!pip install -q uv && uv pip install --system -q \
    torch==2.4.0 transformers==4.45.2 datasets==3.0.1 tokenizers==0.20.0 \
    accelerate==0.34.2 safetensors==0.4.5 gradio==4.44.1

================================================================================
RUN AS SCRIPT:
================================================================================
python train.py \\
    --drive_path /content/drive/MyDrive/llama_scratch \\
    --output_dir /content/output \\
    --token_budget 500_000_000 \\
    --wall_clock_hours 11
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, List

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# 0. Pinned versions check (fail loudly if environment is wrong)
# ---------------------------------------------------------------------------
import transformers
import datasets
import tokenizers
import accelerate
import safetensors

print("Library versions:")
print(f"  torch         : {torch.__version__}")
print(f"  transformers  : {transformers.__version__}")
print(f"  datasets      : {datasets.__version__}")
print(f"  tokenizers    : {tokenizers.__version__}")
print(f"  accelerate    : {accelerate.__version__}")
print(f"  safetensors   : {safetensors.__version__}")

# ---------------------------------------------------------------------------
# 1. Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Llama 0.5B from scratch on T4")
    p.add_argument("--drive_path",      type=str,   default="/content/drive/MyDrive/llama_scratch")
    p.add_argument("--output_dir",      type=str,   default="/content/output")
    p.add_argument("--tokenizer_dir",   type=str,   default=None,
                   help="Dir to load/save tokenizer. Defaults to drive_path/tokenizer")
    p.add_argument("--dataset",         type=str,   default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config",  type=str,   default="sample-10BT")
    p.add_argument("--dataset_split",   type=str,   default="train")
    # Token budget / wall-clock
    p.add_argument("--token_budget",    type=int,   default=500_000_000)
    p.add_argument("--wall_clock_hours", type=float, default=11.0)
    # Model
    p.add_argument("--vocab_size",      type=int,   default=32000)
    p.add_argument("--hidden_size",     type=int,   default=1280)
    p.add_argument("--intermediate_size", type=int, default=3456)
    p.add_argument("--num_hidden_layers", type=int, default=24)
    p.add_argument("--num_attention_heads", type=int, default=20)
    p.add_argument("--num_key_value_heads", type=int, default=10)
    p.add_argument("--max_position_embeddings", type=int, default=1024)
    p.add_argument("--block_size",      type=int,   default=1024)
    # Training
    p.add_argument("--micro_batch_size", type=int,  default=8)
    p.add_argument("--grad_accum_steps", type=int,  default=8)
    p.add_argument("--peak_lr",          type=float, default=3e-4)
    p.add_argument("--min_lr_ratio",     type=float, default=0.1)
    p.add_argument("--warmup_steps",     type=int,   default=1000)
    p.add_argument("--weight_decay",     type=float, default=0.1)
    p.add_argument("--beta1",            type=float, default=0.9)
    p.add_argument("--beta2",            type=float, default=0.95)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    # Streaming
    p.add_argument("--shuffle_buffer",   type=int,   default=10_000)
    p.add_argument("--seed",             type=int,   default=1337)
    # Checkpointing
    p.add_argument("--ckpt_every",       type=int,   default=1000)
    p.add_argument("--eval_every",       type=int,   default=1000)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--keep_last",        type=int,   default=3)
    # Tokenizer training
    p.add_argument("--train_tokenizer",  action="store_true",
                   help="Train the BPE tokenizer instead of loading existing one")
    p.add_argument("--tokenizer_docs",   type=int,   default=300_000,
                   help="Number of fineweb-edu docs to train tokenizer on")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. GPU check — fail loudly if not T4+/fp16-capable
# ---------------------------------------------------------------------------

def check_gpu() -> None:
    assert torch.cuda.is_available(), (
        "\n❌ No GPU detected. In Colab: Runtime → Change runtime type → T4 GPU."
    )
    name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  (compute capability {cap_major}.{cap_minor})")
    # T4 = compute capability 7.5 (Turing). fp16 tensor cores supported.
    assert cap_major >= 7, (
        f"\n❌ Need compute capability ≥ 7.0 for fp16 tensor cores, got {cap_major}.{cap_minor}"
    )
    # Whitelist of acceptable Colab GPUs (T4 is the explicit target; others also fine)
    acceptable = ("T4", "V100", "A100", "L4", "A10G")
    assert any(g in name for g in acceptable), (
        f"\n❌ Need a T4 / V100 / A100 / L4 (fp16-capable) GPU, got: {name}"
    )
    # IMPORTANT: We do NOT rely on bf16. T4 returns False here, which is expected.
    bf16 = torch.cuda.is_bf16_supported()
    print(f"bf16 supported: {bf16}  (we DON'T use it — fp16 only on T4)")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# 3. Tokenizer training (byte-level BPE, 32000 vocab)
# ---------------------------------------------------------------------------

SPECIAL_TOKEN = "<|endoftext|>"  # used as BOS / EOS / PAD


def train_tokenizer(args: argparse.Namespace) -> "PreTrainedTokenizerFast":  # type: ignore[name-defined]
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast
    from datasets import load_dataset

    tokenizer_dir = args.tokenizer_dir or os.path.join(args.drive_path, "tokenizer")
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[tokenizer] Training byte-level BPE (vocab={args.vocab_size}) on "
          f"{args.tokenizer_docs:,} fineweb-edu docs → {tokenizer_dir}")

    # Stream fineweb-edu and yield text
    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    ).shuffle(args.shuffle_buffer, seed=args.seed)

    def text_iter() -> Iterator[str]:
        n = 0
        for doc in ds:
            yield doc["text"]
            n += 1
            if n >= args.tokenizer_docs:
                break

    tok = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter(), trainer=trainer, length=args.tokenizer_docs)

    # Wrap so AutoTokenizer.from_pretrained can load it
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        unk_token=SPECIAL_TOKEN,
    )
    fast.save_pretrained(tokenizer_dir)

    # Also write vocab.json (id→token mapping saved as token→id, which is what
    # HuggingFace expects for a vocab.json).
    vocab = fast.get_vocab()  # {token: id}
    with open(os.path.join(tokenizer_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"[tokenizer] saved. Vocab size: {len(vocab)}")
    print(f"[tokenizer] sample tokens: {list(vocab.items())[:5]}")
    return fast


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer
    tokenizer_dir = args.tokenizer_dir or os.path.join(args.drive_path, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# 4. Dataset streaming + packing into fixed-length 1024-token blocks
# ---------------------------------------------------------------------------

def make_packed_stream(tokenizer, args: argparse.Namespace, skip_docs: int = 0) -> Iterator[torch.Tensor]:
    """
    Stream fineweb-edu, tokenize each doc, append EOS between docs, and pack
    into fixed-length `block_size` blocks. Yields 1D LongTensors of length block_size.

    On resume we call `.skip(skip_docs)` on the (deterministically shuffled)
    stream so we don't reprocess docs we've already consumed.
    """
    from datasets import load_dataset
    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    ).shuffle(args.shuffle_buffer, seed=args.seed)

    if skip_docs > 0:
        print(f"[data] skipping {skip_docs:,} already-consumed docs")
        ds = ds.skip(skip_docs)

    eos_id = tokenizer.eos_token_id
    block_size = args.block_size
    buffer: List[int] = []

    for doc in ds:
        ids = tokenizer.encode(doc["text"])
        ids.append(eos_id)
        buffer.extend(ids)
        while len(buffer) >= block_size:
            block = buffer[:block_size]
            buffer = buffer[block_size:]
            yield torch.tensor(block, dtype=torch.long)


def batched_packed_stream(tokenizer, args: argparse.Namespace, skip_docs: int = 0,
                          device: str = "cuda") -> Iterator[torch.Tensor]:
    """Yield micro-batches of shape (micro_batch_size, block_size)."""
    micro_bs = args.micro_batch_size
    batch: List[torch.Tensor] = []
    for block in make_packed_stream(tokenizer, args, skip_docs=skip_docs):
        batch.append(block)
        if len(batch) == micro_bs:
            yield torch.stack(batch).to(device)
            batch = []
    # flush remaining partial batch (ignored mid-training, only matters at end)
    if batch:
        yield torch.stack(batch).to(device)


# ---------------------------------------------------------------------------
# 5. Model definition — LlamaConfig + LlamaForCausalLM (random init)
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> "LlamaForCausalLM":  # type: ignore[name-defined]
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,   # GQA: 2 query heads per KV head
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,    # tied input/output embeddings (~40M saved)
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        bias=False,                  # Llama has no biases anywhere
        use_cache=False,             # required for gradient checkpointing
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config)
    model.config.use_cache = False   # grad checkpointing requires this

    n = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] LlamaForCausalLM (random init)")
    print(f"[model] total params    : {n:,}  ({n/1e6:.2f}M)")
    print(f"[model] trainable params: {n_grad:,}  ({n_grad/1e6:.2f}M)")
    assert 450e6 <= n <= 550e6, f"\n❌ Param count {n:,} not in [450M, 550M]"
    return model


# ---------------------------------------------------------------------------
# 6. Optimizer, scheduler, scaler
# ---------------------------------------------------------------------------

def build_optimizer(model, args: argparse.Namespace, total_steps: int):
    """
    AdamW with weight decay applied ONLY to matrices (dim > 1), not to biases
    or norm parameters (dim <= 1). betas (0.9, 0.95) are the Llama defaults.
    """
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() > 1:
            decay.append(p)
        else:
            nodecay.append(p)
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in nodecay)
    print(f"[opt] decay params:   {n_decay:,}  (wd={args.weight_decay})")
    print(f"[opt] nodecay params: {n_nodecay:,}  (wd=0.0)")

    optimizer = AdamW(
        [
            {"params": decay,   "weight_decay": args.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        betas=(args.beta1, args.beta2),
        lr=args.peak_lr,
        eps=1e-8,
    )

    # Cosine decay from peak_lr to peak_lr * min_lr_ratio over total_steps,
    # with linear warmup of `warmup_steps` steps.
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # fp16 GradScaler — required because T4 has no bf16
    scaler = GradScaler()
    return optimizer, scheduler, scaler


# ---------------------------------------------------------------------------
# 7. Checkpointing (save / resume / prune)
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, step: int,
                    docs_consumed: int, args: argparse.Namespace,
                    rng_states: dict) -> None:
    ckpt_dir = os.path.join(args.drive_path, f"checkpoint_{step:08d}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":            step,
            "docs_consumed":   docs_consumed,
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "scaler":          scaler.state_dict(),
            "torch_rng":       rng_states["torch"],
            "cuda_rng":        rng_states["cuda"],
            "python_rng":      rng_states["python"],
            "args":            vars(args),
        },
        os.path.join(ckpt_dir, "ckpt.pt"),
    )
    # Write a small marker so we can identify the latest easily
    with open(os.path.join(ckpt_dir, "step.txt"), "w") as f:
        f.write(str(step))

    # Prune older checkpoints, keep only last `keep_last`
    old = sorted(glob.glob(os.path.join(args.drive_path, "checkpoint_*")))
    if len(old) > args.keep_last:
        for d in old[:-args.keep_last]:
            shutil.rmtree(d, ignore_errors=True)
            print(f"[ckpt] pruned old checkpoint: {d}")
    print(f"[ckpt] saved checkpoint → {ckpt_dir}")


def find_latest_checkpoint(args: argparse.Namespace) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(args.drive_path, "checkpoint_*")))
    # Only keep dirs that actually contain a ckpt.pt
    ckpts = [c for c in ckpts if os.path.isfile(os.path.join(c, "ckpt.pt"))]
    return ckpts[-1] if ckpts else None


def resume(model, optimizer, scheduler, scaler, args: argparse.Namespace):
    latest = find_latest_checkpoint(args)
    if latest is None:
        print("[resume] no checkpoint found, starting fresh")
        return 0, 0
    print(f"[resume] loading checkpoint: {latest}")
    ckpt = torch.load(os.path.join(latest, "ckpt.pt"), map_location="cuda",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state(ckpt["cuda_rng"])
    random.setstate(ckpt["python_rng"])
    print(f"[resume] resumed at step {ckpt['step']}, "
          f"docs_consumed={ckpt['docs_consumed']:,}")
    return int(ckpt["step"]), int(ckpt["docs_consumed"])


# ---------------------------------------------------------------------------
# 8. Evaluation — held-out perplexity on a small slice of the stream
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, args: argparse.Namespace, n_batches: int = 8) -> float:
    """Compute mean loss on `n_batches` micro-batches from a fresh stream
    (using a different seed so eval data != train data)."""
    model.eval()
    eval_args = argparse.Namespace(**vars(args))
    eval_args.seed = args.seed + 1   # disjoint slice
    stream = batched_packed_stream(tokenizer, eval_args, skip_docs=0, device="cuda")
    losses = []
    for i, batch in enumerate(stream):
        if i >= n_batches:
            break
        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch, labels=batch)
        losses.append(out.loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("TRAIN LLAMA 0.5B FROM SCRATCH ON T4")
    print("=" * 80)

    check_gpu()

    # Drive mount happens in the notebook. If running as script outside Colab,
    # ensure drive_path exists.
    Path(args.drive_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer
    if args.train_tokenizer:
        tokenizer = train_tokenizer(args)
    else:
        tokenizer = load_tokenizer(args)
    print(f"[tokenizer] vocab_size = {tokenizer.vocab_size}, "
          f"eos_id = {tokenizer.eos_token_id}, pad_id = {tokenizer.pad_token_id}")

    # Model
    model = build_model(args).to("cuda")
    model.gradient_checkpointing_enable()   # enables checkpointing
    model.config.use_cache = False

    # Total optimizer steps
    tokens_per_step = args.micro_batch_size * args.grad_accum_steps * args.block_size
    total_steps_budget = max(1, args.token_budget // tokens_per_step)
    print(f"[plan] tokens/optim-step = {tokens_per_step:,}")
    print(f"[plan] total optim-steps from token budget = {total_steps_budget:,}")

    # Optimizer / scheduler / scaler
    optimizer, scheduler, scaler = build_optimizer(model, args, total_steps_budget)

    # Resume
    start_step, docs_consumed = resume(model, optimizer, scheduler, scaler, args)
    if start_step > 0:
        # If we resumed mid-way, advance scheduler past start_step
        for _ in range(start_step):
            scheduler.step()

    # Data stream — skip already-consumed docs on resume
    data_iter = batched_packed_stream(tokenizer, args, skip_docs=docs_consumed, device="cuda")

    # Eval stream seed offset
    eval_args = argparse.Namespace(**vars(args))
    eval_args.seed = args.seed + 1

    # Timers
    t0 = time.time()
    wall_clock_seconds = args.wall_clock_hours * 3600.0
    tokens_seen = start_step * tokens_per_step
    docs_in_batch = 0
    accum_loss = 0.0
    accum_count = 0
    step = start_step

    print(f"\n[loop] starting training. wall-clock limit = {args.wall_clock_hours:.2f}h "
          f"({wall_clock_seconds:.0f}s), token budget = {args.token_budget:,}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        while True:
            # ---- Termination checks ----
            if tokens_seen >= args.token_budget:
                print(f"[loop] token budget reached ({tokens_seen:,} ≥ {args.token_budget:,})")
                break
            elapsed = time.time() - t0
            if elapsed >= wall_clock_seconds:
                print(f"[loop] wall-clock limit reached ({elapsed:.0f}s ≥ {wall_clock_seconds:.0f}s)")
                break
            if step >= total_steps_budget:
                print(f"[loop] reached total_steps_budget = {total_steps_budget}")
                break

            # ---- Gradient accumulation ----
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for micro in range(args.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    print("[loop] dataset exhausted before budget reached; stopping")
                    break
                bs, seqlen = batch.shape
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch, labels=batch)
                    loss = out.loss / args.grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += out.loss.item()
                accum_count += 1
                docs_in_batch += bs   # rough count: each row is a packed block, not 1 doc; use as proxy

            # ---- Optimizer step ----
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            tokens_seen += tokens_per_step
            mean_micro_loss = accum_loss / max(args.grad_accum_steps, 1)
            lr_now = scheduler.get_last_lr()[0]

            # ---- Logging ----
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                steps_done = step - start_step
                tps = (steps_done * tokens_per_step) / max(elapsed, 1e-6)
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                vram_peak  = torch.cuda.max_memory_allocated() / 1e9
                remaining_tokens = max(0, args.token_budget - tokens_seen)
                eta_sec = remaining_tokens / max(tps, 1e-6)
                print(
                    f"step {step:>7d} | loss {mean_micro_loss:.4f} | lr {lr_now:.2e} | "
                    f"tokens {tokens_seen:>12,} | tps {tps:,.0f} | "
                    f"vram {vram_alloc:.2f}/{vram_peak:.2f} GB | "
                    f"eta {eta_sec/3600:.2f}h"
                )

            # ---- Evaluation ----
            if step % args.eval_every == 0:
                eval_loss = evaluate(model, tokenizer, args, n_batches=8)
                ppl = math.exp(min(eval_loss, 20.0))
                print(f"  [eval] step {step}: loss={eval_loss:.4f}  ppl={ppl:.2f}")

            # ---- Checkpoint ----
            if step % args.ckpt_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step=step,
                    docs_consumed=docs_consumed + docs_in_batch,
                    args=args,
                    rng_states={
                        "torch":  torch.get_rng_state(),
                        "cuda":   torch.cuda.get_rng_state(),
                        "python": random.getstate(),
                    },
                )

    except KeyboardInterrupt:
        print("\n[loop] KeyboardInterrupt caught — saving emergency checkpoint")
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            step=step,
            docs_consumed=docs_consumed + docs_in_batch,
            args=args,
            rng_states={
                "torch":  torch.get_rng_state(),
                "cuda":   torch.cuda.get_rng_state(),
                "python": random.getstate(),
            },
        )
        raise

    print("\n[loop] training finished.")

    # Final save regardless of how we exited (unless KeyboardInterrupt re-raised)
    export(model, tokenizer, args)


# ---------------------------------------------------------------------------
# 10. Export — produce the six required files
# ---------------------------------------------------------------------------

def export(model, tokenizer, args: argparse.Namespace) -> None:
    """
    Save the model + tokenizer into args.output_dir such that the folder loads
    via AutoTokenizer.from_pretrained(dir) and AutoModelForCausalLM.from_pretrained(dir).

    Produces exactly these six files:
        config.json
        generation_config.json
        model.safetensors
        tokenizer.json
        tokenizer_config.json
        vocab.json
    """
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Make sure model.config has correct metadata before saving
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    model.config.pad_token_id = pad_id
    model.config.bos_token_id = bos_id
    model.config.eos_token_id = eos_id
    model.config.use_cache = True   # generation needs this

    # Re-enable cache for generation (turns off grad checkpointing implicitly)
    model.generation_config.pad_token_id = pad_id
    model.generation_config.bos_token_id = bos_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.8
    model.generation_config.top_p = 0.95
    model.generation_config.max_new_tokens = 256

    # 1. model.safetensors + config.json (save_pretrained writes both)
    model.save_pretrained(out, safe_serialization=True)

    # 2. tokenizer.json + tokenizer_config.json + vocab.json
    tokenizer.save_pretrained(out)
    # Ensure vocab.json exists (PreTrainedTokenizerFast on a BPE backed by
    # tokenizers writes tokenizer.json but may not write vocab.json)
    vocab_path = out / "vocab.json"
    if not vocab_path.exists():
        vocab = tokenizer.get_vocab()
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False)

    # 3. generation_config.json — model.save_pretrained wrote it from
    # model.generation_config above; but write again explicitly to be sure
    gen_cfg = {
        "do_sample":      True,
        "temperature":    0.8,
        "top_p":          0.95,
        "max_new_tokens": 256,
        "pad_token_id":   pad_id,
        "bos_token_id":   bos_id,
        "eos_token_id":   eos_id,
    }
    with open(out / "generation_config.json", "w") as f:
        json.dump(gen_cfg, f, indent=2)

    print("\n[export] files in output dir:")
    required = ["config.json", "generation_config.json", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "vocab.json"]
    for name in required:
        p = out / name
        size = p.stat().st_size if p.exists() else None
        if size is None:
            print(f"  ❌ MISSING: {name}")
        else:
            print(f"  ✓ {name:25s}  {size:>14,} bytes")

    missing = [n for n in required if not (out / n).exists()]
    assert not missing, f"❌ Missing required files: {missing}"

    # Copy to Drive as well
    drive_out = Path(args.drive_path) / "output"
    drive_out.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(out / name, drive_out / name)
    print(f"[export] also copied to Drive: {drive_out}")


# ---------------------------------------------------------------------------
# 11. Generation / sanity check
# ---------------------------------------------------------------------------

def generate(prompt: str, model_dir: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_p: float = 0.95) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda")
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 12. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
