---
Task ID: 1
Agent: main
Task: Build a complete runnable Colab notebook + train.py that trains a ~0.5B Llama-style decoder-only LM from scratch on T4.

Work Log:
- Created `/home/z/my-project/download/` and `/home/z/my-project/scripts/`.
- Wrote `/home/z/my-project/download/train.py` (748 lines, standalone training script with argparse).
- Wrote `/home/z/my-project/scripts/build_notebook.py` (notebook generator with all 8 sections).
- Ran generator → produced `/home/z/my-project/download/train_llama_from_scratch.ipynb` (25 cells, 41.7 KB).
- Validated notebook JSON (nbformat 4.0, accelerator=GPU).
- AST-parsed all 15 code cells (only the install cell with `!` shell magic is non-AST, expected for Colab).
- Ran 59-point requirements checklist against the notebook — all PASS (initial 14 "failures" were false negatives from too-strict whitespace matching in form cells; verified via whitespace-insensitive re-check that all hyperparameters are present).
- Verified param count math: ~477.5M (embed 40.96M + 24 layers * 18.19M + final norm 1.28k) → within [450M, 550M].

Stage Summary:
- Deliverables in `/home/z/my-project/download/`:
  - `train_llama_from_scratch.ipynb` (25 cells, 8 labeled sections, Colab `#@param` forms for DRIVE_PATH/TOKEN_BUDGET/WALL_CLOCK_HOURS)
  - `train.py` (standalone script equivalent; pip install line in module docstring)
- Architecture: LlamaConfig(hidden=1280, layers=24, heads=20, kv_heads=10, inter=3456, vocab=32000, max_pos=1024, tied embeddings, SDPA attention, no biases), ~477.5M params.
- Training: fp16 autocast + GradScaler (no bf16), grad checkpointing, micro_bs=8 × accum=8 × seq=1024 = 65,536 tokens/step, AdamW (β=0.9,0.95, wd=0.1 matrices only), peak LR 3e-4, linear 1k warmup → cosine decay to 10%.
- Data: HuggingFaceFW/fineweb-edu sample-10BT, streaming=True, shuffle buffer 10k, seed 1337, packed into 1024-token blocks with EOS between docs.
- Tokenizer: byte-level BPE trained on 300k streamed docs, vocab 32000, single `` token as BOS/EOS/PAD/UNK. Saves tokenizer.json, tokenizer_config.json, vocab.json explicitly.
- Checkpointing: every 1000 steps to Drive, saves model+opt+scaler+sched+step+RNG+docs_consumed, keeps last 3, auto-resume from latest, KeyboardInterrupt → save+re-raise.
- Export: writes the exact 6 required files (config.json, generation_config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.json) to /content/output AND Drive.
- Verification cell: loads via AutoTokenizer.from_pretrained + AutoModelForCausalLM.from_pretrained and generates text.
- Gradio interface behind USE_GRADIO toggle.
- Pinned versions: torch==2.4.0, transformers==4.45.2, datasets==3.0.1, tokenizers==0.20.0, accelerate==0.34.2, safetensors==0.4.5, gradio==4.44.1.
- Install via `!pip install -q uv && uv pip install --system -q ...`.
