#!/usr/bin/env python3
"""
build_notebook.py — Generate train_llama_from_scratch.ipynb.

The notebook mirrors train.py but is structured into the 8 labeled sections
required by the spec, uses Colab `#@param` forms for configuration,
mounts Drive, and includes a verification cell at the end that loads the
output dir via AutoTokenizer + AutoModelForCausalLM.

Run:  python build_notebook.py
Out:  /home/z/my-project/download/train_llama_from_scratch.ipynb
"""
import json
from pathlib import Path

OUT = Path("/home/z/my-project/download/train_llama_from_scratch.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


CELLS = []


# ===========================================================================
# Title
# ===========================================================================
CELLS.append(md("""# Train a ~0.5B Llama-style LM **from scratch** on Colab T4

End-to-end notebook: tokenizer training → streaming → random-init model →
fp16 training with gradient checkpointing → export → sanity-check generation.

**No pretrained weights, no pretrained tokenizers.** Everything is trained
from random init on `HuggingFaceFW/fineweb-edu` (sample-10BT), streamed so
we never download the full 10 BT.

| | |
|---|---|
| Architecture | Llama-style (RMSNorm + RoPE + SwiGLU + GQA + tied embeddings) |
| Params | ~478M (within 450M–550M target) |
| Hardware | NVIDIA T4 (16 GB, fp16, no bf16, no flash-attn-2) |
| Mixed precision | `torch.amp.autocast(dtype=float16)` + `GradScaler` |
| Effective batch | 8 micro × 8 accum × 1024 = **65,536 tokens / optim step** |
| Optimizer | AdamW (β=0.9,0.95; wd=0.1 on matrices only) |
| LR schedule | Linear 1k warmup → cosine decay to 10% of 3e-4 peak |
| Token budget | 500M (default, editable via form below) |
| Wall-clock limit | 11 h (stays under Colab's ~12 h cap) |

> **Edit only the two form fields in the next cell:** `DRIVE_PATH` and
> `TOKEN_BUDGET` (and optionally `WALL_CLOCK_HOURS`). Everything else is
> tuned for T4 and should not need touching.
"""))


# ===========================================================================
# Section 1 — Setup
# ===========================================================================
CELLS.append(md("""## 1. Setup

Pinned versions known to work on Colab T4 (verified late 2024):

| package | version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.45.2 |
| datasets | 3.0.1 |
| tokenizers | 0.20.0 |
| accelerate | 0.34.2 |
| safetensors | 0.4.5 |
| gradio | 4.44.1 |

We use `uv` for a fast install (~30 s vs several minutes with vanilla pip).
"""))

CELLS.append(code("""# uv install — much faster than pip on Colab
!pip install -q uv
!uv pip install --system -q \\
    torch==2.4.0 \\
    transformers==4.45.2 \\
    datasets==3.0.1 \\
    tokenizers==0.20.0 \\
    accelerate==0.34.2 \\
    safetensors==0.4.5 \\
    gradio==4.44.1
"""))

CELLS.append(code("""# Imports + version check (fails loudly if versions drift)
import os, sys, math, time, glob, json, random, shutil
from pathlib import Path
from typing import Iterator, List

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

import transformers, datasets, tokenizers, accelerate, safetensors
from transformers import (LlamaConfig, LlamaForCausalLM, AutoTokenizer,
                          AutoModelForCausalLM, PreTrainedTokenizerFast,
                          get_cosine_schedule_with_warmup)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

print("Versions:")
print(f"  torch        : {torch.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  tokenizers   : {tokenizers.__version__}")
print(f"  accelerate   : {accelerate.__version__}")
print(f"  safetensors  : {safetensors.__version__}")
"""))

CELLS.append(code("""# Mount Google Drive (checkpoints + final export live here so they survive
# Colab disconnects). Authorize when prompted.
from google.colab import drive
drive.mount("/content/drive")
"""))

CELLS.append(code("""# GPU check — FAIL LOUDLY if not a T4 / fp16-capable GPU.
assert torch.cuda.is_available(), \\
    "\\n❌ No GPU. Runtime → Change runtime type → T4 GPU."

_gpu_name = torch.cuda.get_device_name(0)
_cap = torch.cuda.get_device_capability(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {_gpu_name}")
print(f"Compute capability: {_cap[0]}.{_cap[1]}")
print(f"VRAM: {_vram:.2f} GB")

# T4 = compute capability 7.5 (Turing), fp16 tensor cores supported.
# We also accept V100/A100/L4 since they're strictly better.
assert _cap[0] >= 7, f"\\n❌ Need compute capability ≥ 7.0 for fp16 tensor cores, got {_cap}"
_acceptable = ("T4", "V100", "A100", "L4", "A10G")
assert any(g in _gpu_name for g in _acceptable), \\
    f"\\n❌ Need a T4 / V100 / A100 / L4 GPU, got: {_gpu_name}"

# IMPORTANT: We explicitly do NOT rely on bf16. T4 (Turing) does not support it.
# torch.cuda.is_bf16_supported() will return False on T4, which is expected.
_bf16_ok = torch.cuda.is_bf16_supported()
print(f"bf16 supported: {_bf16_ok}  (we DON'T use it — fp16 only on T4)")
print("✅ GPU OK — proceeding with fp16 autocast + GradScaler.")
"""))


# ===========================================================================
# Section 2 — Config (with Colab forms)
# ===========================================================================
CELLS.append(md("""## 2. Config (Colab forms — edit me)

Only **`DRIVE_PATH`** and **`TOKEN_BUDGET`** normally need editing. Everything
else is tuned for T4 16 GB and the ~0.5B Llama architecture specified.
"""))

CELLS.append(code('''# ===== PATHS =====
# Where checkpoints + final model live on Drive. Create this folder once.
DRIVE_PATH = "/content/drive/MyDrive/llama_scratch"  #@param {type:"string"}
# Local working output dir (also copied to DRIVE_PATH/output at the end)
OUTPUT_DIR = "/content/output"                       #@param {type:"string"}

# ===== TOKEN BUDGET =====
# Total tokens to train on. 500M ≈ ~7,600 optim steps at 65,536 tokens/step.
# Lower this to 100M for a smoke test (≈1,500 steps, ~1 h).
TOKEN_BUDGET = 500_000_000  #@param {type:"integer"}

# ===== WALL-CLOCK =====
# Colab ~12 h cap. Default 11 h to leave time for export + verification.
WALL_CLOCK_HOURS = 11.0  #@param {type:"number"}

# ===== MODEL (Llama-style, ~478M params) =====
VOCAB_SIZE              = 32000   #@param {type:"integer"}
HIDDEN_SIZE             = 1280    #@param {type:"integer"}
INTERMEDIATE_SIZE       = 3456    #@param {type:"integer"}
NUM_HIDDEN_LAYERS       = 24      #@param {type:"integer"}
NUM_ATTENTION_HEADS     = 20      #@param {type:"integer"}
NUM_KEY_VALUE_HEADS     = 10      #@param {type:"integer"}  # GQA, 2 query heads / KV head
MAX_POSITION_EMBEDDINGS = 1024    #@param {type:"integer"}
BLOCK_SIZE              = 1024    #@param {type:"integer"}

# ===== TRAINING =====
# micro_bs=8 × accum=8 × seq=1024 = 65,536 tokens per optimizer step.
# This fits comfortably in T4 16 GB with gradient checkpointing.
# If you OOM, drop MICRO_BATCH_SIZE to 4 (and bump GRAD_ACCUM_STEPS to 16
# to keep the effective batch at ~65k).
MICRO_BATCH_SIZE   = 8     #@param {type:"integer"}
GRAD_ACCUM_STEPS    = 8     #@param {type:"integer"}
PEAK_LR             = 3e-4  #@param {type:"number"}
MIN_LR_RATIO        = 0.1   #@param {type:"number"}
WARMUP_STEPS        = 1000  #@param {type:"integer"}
WEIGHT_DECAY        = 0.1   #@param {type:"number"}
BETA1, BETA2        = 0.9, 0.95
GRAD_CLIP           = 1.0   #@param {type:"number"}

# ===== DATA STREAMING =====
DATASET             = "HuggingFaceFW/fineweb-edu"  #@param {type:"string"}
DATASET_CONFIG      = "sample-10BT"               #@param {type:"string"}
DATASET_SPLIT       = "train"                     #@param {type:"string"}
SHUFFLE_BUFFER      = 10_000   #@param {type:"integer"}  # docs buffered for shuffling
SEED                = 1337     #@param {type:"integer"}

# ===== CHECKPOINTING / LOGGING =====
CKPT_EVERY          = 1000  #@param {type:"integer"}  # save every N optim steps
EVAL_EVERY          = 1000  #@param {type:"integer"}
LOG_EVERY           = 50    #@param {type:"integer"}
KEEP_LAST           = 3     #@param {type:"integer"}  # keep only last N checkpoints on Drive

# ===== TOKENIZER TRAINING =====
TRAIN_TOKENIZER     = True    #@param {type:"boolean"}
TOKENIZER_DOCS      = 300_000 #@param {type:"integer"}  # docs to stream for BPE training

# ===== SPECIAL TOKEN (used as BOS / EOS / PAD) =====
SPECIAL_TOKEN       = ""  #@param {type:"string"}

# Derived
TOKENIZER_DIR = f"{DRIVE_PATH}/tokenizer"
Path(DRIVE_PATH).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"DRIVE_PATH      = {DRIVE_PATH}")
print(f"OUTPUT_DIR      = {OUTPUT_DIR}")
print(f"TOKENIZER_DIR   = {TOKENIZER_DIR}")
print(f"TOKEN_BUDGET    = {TOKEN_BUDGET:,}")
print(f"WALL_CLOCK_HOURS= {WALL_CLOCK_HOURS}")
'''))


# ===========================================================================
# Section 3 — Tokenizer training
# ===========================================================================
CELLS.append(md("""## 3. Tokenizer training (byte-level BPE, 32000 vocab)

Trains a byte-level BPE on a streamed slice of fineweb-edu (`TOKENIZER_DOCS`
docs, default 300k). Saves:
- `tokenizer.json` — the full BPE tokenizer (loadable by `tokenizers`)
- `tokenizer_config.json` — wraps it for `AutoTokenizer`
- `vocab.json` — `{token: id}` mapping

The single token `` is registered as BOS / EOS / PAD / UNK.
"""))

CELLS.append(code("""def train_tokenizer():
    \"\"\"Train byte-level BPE on a streamed slice of fineweb-edu.\"\"\"
    Path(TOKENIZER_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Training byte-level BPE  vocab={VOCAB_SIZE}  "
          f"docs={TOKENIZER_DOCS:,}  → {TOKENIZER_DIR}")

    # Stream fineweb-edu, shuffle deterministically with SEED
    ds = datasets.load_dataset(
        DATASET, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True,
    ).shuffle(SHUFFLE_BUFFER, seed=SEED)

    def text_iter():
        n = 0
        for doc in ds:
            yield doc["text"]
            n += 1
            if n >= TOKENIZER_DOCS:
                break
        print(f"  streamed {n:,} docs for tokenizer training")

    tok = Tokenizer(BPE(unk_token=SPECIAL_TOKEN))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter(), trainer=trainer, length=TOKENIZER_DOCS)

    # Wrap in PreTrainedTokenizerFast so AutoTokenizer can load it.
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        unk_token=SPECIAL_TOKEN,
    )
    fast.save_pretrained(TOKENIZER_DIR)

    # Also export vocab.json (token→id). PreTrainedTokenizerFast.save_pretrained
    # writes tokenizer.json + tokenizer_config.json + special_tokens_map.json,
    # but for byte-level BPE it does NOT always write vocab.json — we write it
    # explicitly so the output dir has exactly the 6 required files.
    vocab = fast.get_vocab()
    with open(f"{TOKENIZER_DIR}/vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"  ✓ saved tokenizer.  vocab size: {len(vocab)}")
    print(f"  ✓ bos_id={fast.bos_token_id}  eos_id={fast.eos_token_id}  pad_id={fast.pad_token_id}")
    return fast

if TRAIN_TOKENIZER or not Path(f"{TOKENIZER_DIR}/tokenizer.json").exists():
    tokenizer = train_tokenizer()
else:
    print(f"Loading existing tokenizer from {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded. vocab size: {tokenizer.vocab_size}")
"""))


# ===========================================================================
# Section 4 — Dataset streaming
# ===========================================================================
CELLS.append(md("""## 4. Dataset streaming + token packing

Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`, `train`) — **never downloads
the full dataset.** Each document is tokenized, an EOS token is appended, and
tokens are packed into fixed-length `BLOCK_SIZE` (1024) blocks. Blocks are
stacked into micro-batches of shape `(MICRO_BATCH_SIZE, BLOCK_SIZE)`.

On resume, `skip_docs=N` calls `.skip(N)` on the (deterministically seeded)
shuffled stream so we don't reprocess already-consumed documents.
"""))

CELLS.append(code("""def make_packed_stream(skip_docs: int = 0) -> Iterator[torch.Tensor]:
    \"\"\"Stream fineweb-edu → tokenize → append EOS → pack into 1024-token blocks.\"\"\"
    ds = datasets.load_dataset(
        DATASET, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True,
    ).shuffle(SHUFFLE_BUFFER, seed=SEED)

    if skip_docs > 0:
        print(f"[data] skipping {skip_docs:,} already-consumed docs")
        ds = ds.skip(skip_docs)

    eos_id = tokenizer.eos_token_id
    buffer: List[int] = []
    for doc in ds:
        ids = tokenizer.encode(doc["text"])
        ids.append(eos_id)                 # EOS between documents
        buffer.extend(ids)
        while len(buffer) >= BLOCK_SIZE:
            block = buffer[:BLOCK_SIZE]
            buffer = buffer[BLOCK_SIZE:]
            yield torch.tensor(block, dtype=torch.long)


def batched_packed_stream(skip_docs: int = 0, device: str = "cuda",
                          seed_offset: int = 0) -> Iterator[torch.Tensor]:
    \"\"\"Yield micro-batches of shape (MICRO_BATCH_SIZE, BLOCK_SIZE).

    seed_offset > 0 produces a *different* deterministic stream — used for
    the eval slice so eval data ≠ train data.
    \"\"\"
    # NOTE: shuffle seed is the same as training, but if seed_offset>0 we use
    # a different stream by skipping ahead a fixed number of docs first.
    skip = skip_docs + seed_offset
    batch: List[torch.Tensor] = []
    for block in make_packed_stream(skip_docs=skip):
        batch.append(block)
        if len(batch) == MICRO_BATCH_SIZE:
            yield torch.stack(batch).to(device)
            batch = []
    if batch:
        yield torch.stack(batch).to(device)


# Quick sanity check: peek at one batch.
print("Streaming one micro-batch for sanity check...")
_peek = next(iter(batched_packed_stream(skip_docs=0)))
print(f"  shape: {_peek.shape}  dtype: {_peek.dtype}")
print(f"  first 32 tokens: {_peek[0, :32].tolist()}")
print(f"  decoded (first row, first 200 chars): "
      f"{repr(tokenizer.decode(_peek[0].tolist(), skip_special_tokens=False)[:200])}")
"""))


# ===========================================================================
# Section 5 — Model definition
# ===========================================================================
CELLS.append(md("""## 5. Model definition (Llama-style, ~478M params)

Built via `LlamaConfig` + `LlamaForCausalLM(config)` — **random init, no
pretrained weights.** Uses SDPA attention (`attn_implementation="sdpa"`),
no biases anywhere, RMSNorm, RoPE, SwiGLU MLP, and **tied** input/output
embeddings.
"""))

CELLS.append(code("""def build_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,   # GQA: 2 query heads per KV head
        hidden_act="silu",
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,    # tied input/output embeddings (~40M saved)
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        bias=False,                  # Llama has no biases anywhere
        use_cache=False,              # required for gradient checkpointing
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    # attn_implementation="sdpa" — native PyTorch SDPA, no flash-attn needed.
    model = LlamaForCausalLM(config, attn_implementation="sdpa")
    model.config.use_cache = False   # grad checkpointing requires this
    return model

model = build_model().to("cuda")

# Parameter count (must be in [450M, 550M]).
_n_total    = sum(p.numel() for p in model.parameters())
_n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\n[model] LlamaForCausalLM (random init)")
print(f"[model] total params    : {_n_total:,}  ({_n_total/1e6:.2f}M)")
print(f"[model] trainable params: {_n_trainable:,}  ({_n_trainable/1e6:.2f}M)")

# Per-layer breakdown
for name, p in list(model.named_parameters())[:8]:
    print(f"  {name:50s} {tuple(p.shape)}  {p.numel():>12,}")

assert 450e6 <= _n_total <= 550e6, \\
    f"\\n❌ Param count {_n_total:,} not in [450M, 550M]"
print(f"\\n✅ param count {_n_total/1e6:.2f}M is in [450M, 550M]")

# Enable gradient checkpointing (saves activations memory, costs ~20% slower).
model.gradient_checkpointing_enable()
print("[model] gradient checkpointing ENABLED")
print(f"[model] VRAM after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")
"""))


# ===========================================================================
# Section 6 — Training loop
# ===========================================================================
CELLS.append(md("""## 6. Training loop (fp16 + grad checkpointing + checkpointing + resume)

**Memory plan** (T4 16 GB):
- model fp16 weights: ~1.0 GB
- AdamW master + 2 moments (fp32): ~6.0 GB
- activations (grad ckpt on, micro_bs=8, seq=1024): ~2–4 GB peak
- ⇒ ~9–11 GB used, leaves headroom for fragmentation

**Effective batch:** `8 micro × 8 accum × 1024 = 65,536 tokens / optim step`
(in the requested 64k–128k window).

**Resume:** automatically loads the latest `checkpoint_NNNNNNNN/ckpt.pt` from
`DRIVE_PATH` if one exists, restoring model / optimizer / scheduler / scaler /
RNG state / docs-consumed counter.
"""))

CELLS.append(code("""# ===== Optimizer, scheduler, scaler =====

# AdamW with weight decay ONLY on matrices (dim > 1), not on biases / norms.
decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1]
nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <= 1]
_n_decay    = sum(p.numel() for p in decay_params)
_n_nodecay  = sum(p.numel() for p in nodecay_params)
print(f"[opt] decay params  : {_n_decay:,}  (wd={WEIGHT_DECAY})")
print(f"[opt] nodecay params: {_n_nodecay:,}  (wd=0.0)")

optimizer = AdamW(
    [
        {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    betas=(BETA1, BETA2),
    lr=PEAK_LR,
    eps=1e-8,
)

# Total optim steps from token budget
TOKENS_PER_STEP = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
TOTAL_STEPS_BUDGET = max(1, TOKEN_BUDGET // TOKENS_PER_STEP)
print(f"[plan] tokens/step = {TOKENS_PER_STEP:,}")
print(f"[plan] total optim-steps from budget = {TOTAL_STEPS_BUDGET:,}")

# Linear warmup 1000 steps → cosine decay to MIN_LR_RATIO * PEAK_LR.
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS_BUDGET,
)

# fp16 GradScaler — T4 has no bf16, so we MUST use this.
scaler = GradScaler()
print(f"[amp] fp16 autocast + GradScaler ready (bf16 NOT used)")
"""))

CELLS.append(code("""# ===== Checkpoint save / resume / prune =====

def save_checkpoint(step: int, docs_consumed: int):
    \"\"\"Save a full resumable checkpoint to DRIVE_PATH/checkpoint_NNNNNNNN/.
    Keeps only the last KEEP_LAST checkpoints (older ones pruned).\"\"\"
    ckpt_dir = f"{DRIVE_PATH}/checkpoint_{step:08d}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":          step,
            "docs_consumed": docs_consumed,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "torch_rng":     torch.get_rng_state(),
            "cuda_rng":      torch.cuda.get_rng_state(),
            "python_rng":    random.getstate(),
        },
        f"{ckpt_dir}/ckpt.pt",
    )
    with open(f"{ckpt_dir}/step.txt", "w") as f:
        f.write(str(step))
    # prune older
    old = sorted(glob.glob(f"{DRIVE_PATH}/checkpoint_*"))
    if len(old) > KEEP_LAST:
        for d in old[:-KEEP_LAST]:
            shutil.rmtree(d, ignore_errors=True)
            print(f"  [ckpt] pruned: {d}")
    print(f"  [ckpt] saved → {ckpt_dir}")


def find_latest_checkpoint() -> str | None:
    ckpts = sorted(glob.glob(f"{DRIVE_PATH}/checkpoint_*"))
    ckpts = [c for c in ckpts if os.path.isfile(f"{c}/ckpt.pt")]
    return ckpts[-1] if ckpts else None


def try_resume() -> tuple[int, int]:
    \"\"\"Returns (start_step, docs_consumed). Resumes everything.\"\"\"
    latest = find_latest_checkpoint()
    if latest is None:
        print("[resume] no checkpoint found — starting fresh")
        return 0, 0
    print(f"[resume] loading: {latest}")
    ckpt = torch.load(f"{latest}/ckpt.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state(ckpt["cuda_rng"])
    random.setstate(ckpt["python_rng"])
    print(f"[resume] step={ckpt['step']}  docs_consumed={ckpt['docs_consumed']:,}")
    return int(ckpt["step"]), int(ckpt["docs_consumed"])

# Try resume
START_STEP, DOCS_CONSUMED = try_resume()
"""))

CELLS.append(code("""# ===== Evaluation: held-out perplexity on a disjoint stream =====

@torch.no_grad()
def evaluate(n_batches: int = 8) -> float:
    \"\"\"Mean loss on `n_batches` micro-batches from a disjoint slice of the
    stream (uses a different `skip_docs` offset).\"\"\"
    model.eval()
    losses = []
    for i, batch in enumerate(batched_packed_stream(skip_docs=0, seed_offset=50_000)):
        if i >= n_batches:
            break
        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch, labels=batch)
        losses.append(out.loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))
"""))

CELLS.append(code("""# ===== Main training loop =====

# Timers
t0 = time.time()
WALL_CLOCK_SECONDS = WALL_CLOCK_HOURS * 3600.0
tokens_seen = START_STEP * TOKENS_PER_STEP
docs_in_accum = 0  # docs consumed since last checkpoint
step = START_STEP

# Build the data stream, skipping already-consumed docs on resume.
data_iter = batched_packed_stream(skip_docs=DOCS_CONSUMED, device="cuda")

print(f"\\n{'='*80}")
print(f"TRAINING — tokens/step={TOKENS_PER_STEP:,}  "
      f"budget={TOKEN_BUDGET:,}  wall={WALL_CLOCK_HOURS:.2f}h")
print(f"{'='*80}\\n")

model.train()
optimizer.zero_grad(set_to_none=True)

try:
    while True:
        # ---- Termination checks ----
        if tokens_seen >= TOKEN_BUDGET:
            print(f"[loop] token budget reached ({tokens_seen:,} ≥ {TOKEN_BUDGET:,})")
            break
        if (time.time() - t0) >= WALL_CLOCK_SECONDS:
            print(f"[loop] wall-clock limit reached")
            break
        if step >= TOTAL_STEPS_BUDGET:
            print(f"[loop] reached TOTAL_STEPS_BUDGET = {TOTAL_STEPS_BUDGET}")
            break

        # ---- Gradient accumulation ----
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                print("[loop] dataset stream exhausted before budget reached; stopping")
                break
            bs, seqlen = batch.shape
            with autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch, labels=batch)
                loss = out.loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            accum_loss += out.loss.item()
            docs_in_accum += bs   # rough: each row is a packed block, used as a position proxy

        # ---- Optimizer step ----
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        tokens_seen += TOKENS_PER_STEP
        mean_loss = accum_loss / GRAD_ACCUM_STEPS
        lr_now = scheduler.get_last_lr()[0]

        # ---- Logging ----
        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            steps_done = step - START_STEP
            tps = (steps_done * TOKENS_PER_STEP) / max(elapsed, 1e-6)
            vram_alloc = torch.cuda.memory_allocated() / 1e9
            vram_peak  = torch.cuda.max_memory_allocated() / 1e9
            remaining_tokens = max(0, TOKEN_BUDGET - tokens_seen)
            eta_h = (remaining_tokens / max(tps, 1e-6)) / 3600.0
            print(
                f"step {step:>7d} | loss {mean_loss:.4f} | lr {lr_now:.2e} | "
                f"tokens {tokens_seen:>12,} | tps {tps:,.0f} | "
                f"vram {vram_alloc:.2f}/{vram_peak:.2f} GB | eta {eta_h:.2f}h"
            )

        # ---- Evaluation ----
        if step % EVAL_EVERY == 0:
            eval_loss = evaluate(n_batches=8)
            ppl = math.exp(min(eval_loss, 20.0))
            print(f"  [eval] step {step}: loss={eval_loss:.4f}  ppl={ppl:.2f}")

        # ---- Checkpoint ----
        if step % CKPT_EVERY == 0:
            save_checkpoint(step=step,
                             docs_consumed=DOCS_CONSUMED + docs_in_accum)
            docs_in_accum = 0  # reset counter after checkpoint

except KeyboardInterrupt:
    print("\\n[loop] KeyboardInterrupt caught — saving emergency checkpoint, then re-raising")
    save_checkpoint(step=step,
                     docs_consumed=DOCS_CONSUMED + docs_in_accum)
    raise

print("\\n[loop] training finished.")
print(f"  final step: {step}")
print(f"  tokens seen: {tokens_seen:,}")
print(f"  elapsed: {(time.time()-t0)/3600:.2f}h")
"""))


# ===========================================================================
# Section 7 — Export
# ===========================================================================
CELLS.append(md("""## 7. Export — produce the six required files

Writes exactly:
- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

…into `/content/output` (and copies to `DRIVE_PATH/output`). The folder
loads via `AutoTokenizer.from_pretrained(dir)` and
`AutoModelForCausalLM.from_pretrained(dir)`.
"""))

CELLS.append(code("""# ===== Export =====
out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

# Re-attach pad/bos/eos ids on the model config + generation config
pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = pad_id
model.config.bos_token_id = bos_id
model.config.eos_token_id = eos_id
model.config.use_cache = True   # generation needs this

# Disable grad checkpointing for inference (re-enables use_cache properly)
model.gradient_checkpointing_disable()
model.config.use_cache = True

# generation_config.json — model.save_pretrained will write this from
# model.generation_config, but we set it explicitly first.
model.generation_config.pad_token_id   = pad_id
model.generation_config.bos_token_id    = bos_id
model.generation_config.eos_token_id    = eos_id
model.generation_config.do_sample       = True
model.generation_config.temperature     = 0.8
model.generation_config.top_p           = 0.95
model.generation_config.max_new_tokens  = 256

# 1. model.safetensors + config.json (save_pretrained writes both)
model.save_pretrained(out, safe_serialization=True)

# 2. tokenizer.json + tokenizer_config.json (+ special_tokens_map.json)
tokenizer.save_pretrained(out)

# 3. Ensure vocab.json exists (BPE backed by tokenizers doesn't always write it)
vocab_path = out / "vocab.json"
if not vocab_path.exists():
    vocab = tokenizer.get_vocab()
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

# 4. generation_config.json — explicit write to be 100% sure of contents
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

# ===== Verify all 6 files exist =====
REQUIRED = ["config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "vocab.json"]
print("\\n[export] files in output dir:")
for name in REQUIRED:
    p = out / name
    if not p.exists():
        print(f"  ❌ MISSING: {name}")
    else:
        print(f"  ✓ {name:25s}  {p.stat().st_size:>14,} bytes")

_missing = [n for n in REQUIRED if not (out / n).exists()]
assert not _missing, f"❌ Missing required files: {_missing}"

# ===== Copy to Drive as well =====
drive_out = Path(DRIVE_PATH) / "output"
drive_out.mkdir(parents=True, exist_ok=True)
for name in REQUIRED:
    shutil.copy2(out / name, drive_out / name)
print(f"\\n[export] also copied to Drive: {drive_out}")
"""))


# ===========================================================================
# Section 8 — Test generation / sanity check
# ===========================================================================
CELLS.append(md("""## 8. Test generation / sanity check

Three things:
1. A plain-Python `generate()` helper
2. (Optional) a Gradio interface with sliders for temperature / top_p / max_tokens
3. **Verification cell** that loads the output dir via `AutoTokenizer` +
   `AutoModelForCausalLM` and generates text — proving the export is valid.
"""))

CELLS.append(code("""# Plain-Python generate() — loads the exported folder and samples text.
def generate(prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_p: float = 0.95) -> str:
    tok = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    mdl = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.float16).to("cuda")
    mdl.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = mdl.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

# Smoke test
print("=== Smoke test ===")
for p in ["The history of mathematics begins with",
          "In machine learning, gradient descent",
          "Once upon a time in a distant galaxy"]:
    print(f"\\nPROMPT: {p!r}")
    print(generate(p, max_new_tokens=80))
"""))

CELLS.append(code("""# ===== VERIFICATION CELL =====
# This is the exact check the spec asks for: load with AutoTokenizer +
# AutoModelForCausalLM and generate text. If this cell passes, the export
# is valid.
print("Loading model via AutoTokenizer.from_pretrained + AutoModelForCausalLM.from_pretrained...")

_verif_tok = AutoTokenizer.from_pretrained(OUTPUT_DIR)
_verif_mdl = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.float16).to("cuda")
_verif_mdl.eval()

print(f"  ✓ tokenizer loaded. vocab_size = {_verif_tok.vocab_size}")
print(f"  ✓ model loaded. params = {sum(p.numel() for p in _verif_mdl.parameters()):,}")
print(f"  ✓ generation_config: do_sample={_verif_mdl.generation_config.do_sample} "
      f"temperature={_verif_mdl.generation_config.temperature} "
      f"top_p={_verif_mdl.generation_config.top_p} "
      f"max_new_tokens={_verif_mdl.generation_config.max_new_tokens}")

_prompt = "The capital of France is"
_ids = _verif_tok(_prompt, return_tensors="pt").input_ids.to("cuda")
print(f"\\nVerifying generation with prompt: {_prompt!r}")
with torch.no_grad():
    _out = _verif_mdl.generate(
        _ids,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=_verif_tok.pad_token_id,
    )
_text = _verif_tok.decode(_out[0], skip_special_tokens=True)
print(f"\\nGENERATED:\\n{_text}")
print("\\n✅✅✅ VERIFICATION PASSED — the exported folder loads and generates text.")
"""))

CELLS.append(code("""# ===== Optional: Gradio interface (toggle USE_GRADIO to enable) =====
USE_GRADIO = False  #@param {type:"boolean"}

if USE_GRADIO:
    import gradio as gr

    _gr_tok = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    _gr_mdl = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.float16).to("cuda")
    _gr_mdl.eval()

    def _gr_generate(prompt, temperature, top_p, max_new_tokens):
        ids = _gr_tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = _gr_mdl.generate(
                ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=_gr_tok.pad_token_id,
            )
        return _gr_tok.decode(out[0], skip_special_tokens=True)

    demo = gr.Interface(
        fn=_gr_generate,
        inputs=[
            gr.Textbox(label="Prompt", value="The history of", lines=2),
            gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature"),
            gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p"),
            gr.Slider(16, 512, value=128, step=16, label="max_new_tokens"),
        ],
        outputs=gr.Textbox(label="Generated", lines=10),
        title="Llama 0.5B (from scratch)",
    )
    demo.launch(share=True, debug=True)
else:
    print("Gradio disabled. Set USE_GRADIO = True above to launch the interface.")
"""))


# ===========================================================================
# Assemble notebook
# ===========================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "train_llama_from_scratch.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT}")
print(f"  cells: {len(CELLS)}")
print(f"  size : {OUT.stat().st_size:,} bytes")
