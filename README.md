# llama-0.5b-from-scratch-t4

Train a **~0.5B-parameter Llama-style decoder-only language model FROM SCRATCH**
(random init, no pretrained weights, no pretrained tokenizers) on a single
NVIDIA T4 GPU via Google Colab.

## Why this exists

A clean, fully-commented, end-to-end recipe for training a small Llama from
scratch on free-tier Colab hardware. Everything is tuned for the T4
(16 GB VRAM, Turing, fp16 only — **no bf16, no FlashAttention-2**).

## Files

| File | Description |
|---|---|
| `download/train_llama_from_scratch.ipynb` | Colab notebook — 8 labeled sections, Colab `#@param` forms, runs top-to-bottom |
| `download/train.py` | Standalone equivalent script (same logic, argparse-driven) |
| `scripts/build_notebook.py` | Generator script that produces the notebook |

## Architecture

Llama-style decoder-only transformer, **~477.5M params** (within 450M–550M):

| | |
|---|---|
| `hidden_size` | 1280 |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 20 |
| `num_key_value_heads` | 10 (GQA, 2 query heads / KV head) |
| `intermediate_size` | 3456 |
| `vocab_size` | 32000 |
| `max_position_embeddings` | 1024 |
| `tie_word_embeddings` | True |
| Attention impl | `sdpa` (native PyTorch, no flash-attn) |

## Training recipe

- **Mixed precision:** `torch.amp.autocast(dtype=torch.float16)` + `GradScaler`
- **Gradient checkpointing:** enabled
- **Effective batch:** `8 micro × 8 accum × 1024 = 65,536 tokens / optim step`
- **Optimizer:** AdamW (β = 0.9, 0.95; wd = 0.1 on matrices only)
- **LR:** peak 3e-4, linear 1k warmup → cosine decay to 10% of peak
- **Grad clip:** 1.0
- **Token budget:** 500M (configurable, default)
- **Wall-clock limit:** 11 h (stays under Colab's ~12 h cap)
- **Eval:** held-out perplexity every 1000 steps
- **Checkpointing:** every 1000 steps to Drive (model + optimizer + scaler +
  scheduler + step + RNG state + docs-consumed), keep last 3, auto-resume
- **Disconnect handling:** `KeyboardInterrupt` → save emergency checkpoint → re-raise

## Data

- Source: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), config `sample-10BT`, split `train`
- `streaming=True` — never downloads the full 10 BT
- Shuffle buffer 10 000 docs, fixed seed
- Tokenized on the fly, packed into 1024-token blocks with EOS between docs

## Tokenizer

- Byte-level BPE trained on ~300k streamed fineweb-edu docs
- Vocab size 32 000
- Single token `` as BOS / EOS / PAD / UNK
- Saves `tokenizer.json`, `tokenizer_config.json`, `vocab.json`

## Final output (6 required files)

After training, the export step writes exactly:

```
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
```

…into `/content/output` (and copies to Drive). The folder loads via:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("/content/output")
mdl = AutoModelForCausalLM.from_pretrained("/content/output", torch_dtype="float16")
```

A verification cell at the end of the notebook does exactly this and generates
text to prove the export is valid.

## Quickstart

1. Open `download/train_llama_from_scratch.ipynb` in Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Edit only `DRIVE_PATH` and `TOKEN_BUDGET` in the Config cell (other params are tuned for T4)
4. Run All
5. For a smoke test first: set `TOKEN_BUDGET = 50_000_000` (~770 steps, ~1 h)

## Pinned library versions (known-good on Colab T4)

```
torch==2.4.0
transformers==4.45.2
datasets==3.0.1
tokenizers==0.20.0
accelerate==0.34.2
safetensors==0.4.5
gradio==4.44.1
```

Installed via `uv` for speed:

```bash
!pip install -q uv && uv pip install --system -q \
    torch==2.4.0 transformers==4.45.2 datasets==3.0.1 tokenizers==0.20.0 \
    accelerate==0.34.2 safetensors==0.4.5 gradio==4.44.1
```

## License

MIT — train your own model with it.
