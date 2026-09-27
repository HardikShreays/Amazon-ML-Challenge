"""Stage 8b — fine-tune a transformer as a pair classifier and score the grey-zone pairs. GPU box.

Standalone: needs only torch, transformers, peft, pandas, pyarrow (pip install -r requirements_gpu.txt).
Final submission (v6): intfloat/multilingual-e5-base (MIT, 278M) with `--full` — a full fine-tune with
the 250k-token embedding table frozen, fits a 4 GB RTX 3050:
    python s8_ce_gpu.py train --model intfloat/multilingual-e5-base --data ce/train.parquet --out ce/e5_model --full --lr 3e-5 --bs 32 --max_len 96 --minutes 20
    python s8_ce_gpu.py score --model intfloat/multilingual-e5-base --adapter ce/e5_model --data ce/test.parquet --out ce/test_ce.parquet --max_len 96

Alternative (not used for the final submission): Qwen3 (Apache-2.0) with a 1-logit classification head on
the last token; LoRA on every linear layer, the head trained in full. Input per pair (raw text, no
normalisation — the model learns the noise):
    "<country>\nA: <S1 name> ; <S1 address>\nB: <satellite name> ; <satellite address>"

    # 1 GPU (A10G / L40S): Qwen3-0.6B.  4-8 GPUs: Qwen3-1.7B (same commands, --nproc_per_node=N)
    torchrun --nproc_per_node=1 s8_ce_gpu.py train --model Qwen/Qwen3-0.6B --data ce/train.parquet --out ce/lora --minutes 50
    torchrun --nproc_per_node=1 s8_ce_gpu.py score --model Qwen/Qwen3-0.6B --adapter ce/lora --data ce/tune.parquet --out ce/tune_ce.parquet
    torchrun --nproc_per_node=1 s8_ce_gpu.py score --model Qwen/Qwen3-0.6B --adapter ce/lora --data ce/test.parquet --out ce/test_ce.parquet

`--minutes` caps training time: after 30 steps the throughput is measured and the cosine schedule is
shrunk so it finishes (fully decayed) inside the budget. Add `--limit 20000` to a score run to benchmark.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup():
    """Initialise DDP when launched by torchrun with several GPUs; returns (rank, world size)."""
    ddp = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if ddp:
        dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    return (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)


def log(rank, *msg):
    """Timestamped print from rank 0 only."""
    if rank == 0:
        print(time.strftime('%H:%M:%S'), *msg, flush=True)


def texts(df):
    """Model input text per pair: country, then record A and record B on their own lines."""
    return [f'{c}\nA: {a}\nB: {b}' for a, b, c in zip(df.a, df.b, df.c)]


def tokenize(tok, df, max_len):
    """Token ids of every pair, truncated to max_len."""
    return tok(texts(df), truncation=True, max_length=max_len, add_special_tokens=True)['input_ids']


def collate(ids, pad_id):
    """Right-pad a batch of token id lists; returns (input_ids, attention_mask) on the GPU."""
    n = max(len(x) for x in ids)
    x = torch.full((len(ids), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(ids), n), dtype=torch.long)
    for i, s in enumerate(ids):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = 1
    return x.cuda(non_blocking=True), m.cuda(non_blocking=True)


def compute_dtype():
    """bf16 on Ampere+ (A10G, L40S, A100); fp16 autocast over fp32 weights on T4 / P100 (no bf16)."""
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def base_model(name, tok):
    """Pretrained encoder/decoder with a 1-logit classification head (bf16 where supported)."""
    dt = compute_dtype()
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, torch_dtype=dt if dt == torch.bfloat16 else torch.float32, attn_implementation='sdpa')
    model.config.pad_token_id = tok.pad_token_id
    return model


def get_tok(name):
    """Tokenizer with right padding and a pad token (decoder models reuse EOS)."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def train(args):
    """Fine-tune the pair classifier with BCE on length-bucketed batches inside the --minutes budget; save the model or LoRA adapter to --out."""
    rank, world = setup()
    tok = get_tok(args.model)
    df = pd.read_parquet(args.data)
    df = df.iloc[rank::world].reset_index(drop=True)
    ids = tokenize(tok, df, args.max_len)
    y = df.label.to_numpy(np.float32)
    lens = np.array([len(s) for s in ids])
    log(rank, f'train pairs/rank {len(df):,}  positive {y.mean():.3f}  mean tokens {lens.mean():.1f}')

    # length-bucketed batches: shuffle, sort inside windows of 64 batches, shuffle the batches
    rng = np.random.default_rng(1234 + rank)
    order = rng.permutation(len(df))
    win = args.bs * 64
    batches = []
    for lo in range(0, len(order), win):
        w = order[lo:lo + win]
        w = w[np.argsort(lens[w], kind='stable')]
        batches += [w[i:i + args.bs] for i in range(0, len(w) - args.bs + 1, args.bs)]
    batches = [batches[i] for i in rng.permutation(len(batches))]
    n_steps = len(batches)
    if world > 1:                                   # identical step count on every rank
        t = torch.tensor([n_steps], device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_steps = int(t.item())

    model = base_model(args.model, tok)
    if args.gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    if args.full:                                   # the 250k-token embedding table is 70% of an XLM-R base:
        for n, p in model.named_parameters():       # freezing it keeps a full fine-tune inside 4 GB
            p.requires_grad = 'word_embeddings' not in n
    else:
      from peft import LoraConfig, get_peft_model
      lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type='SEQ_CLS',
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                      modules_to_save=['score'])
      model = get_peft_model(model, lcfg)
    for p in model.parameters():                    # trainable weights in fp32, frozen base in bf16
        if p.requires_grad:
            p.data = p.data.float()
    model.cuda()
    if rank == 0:
        print(f'trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) if world > 1 else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = [n_steps]
    warm = min(100, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total[0] - warm)))))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=compute_dtype() == torch.float16)
    log(rank, f'compute dtype {compute_dtype()}  gpu {torch.cuda.get_device_name()}')
    net.train()
    t0, run = time.time(), 0.0
    step = 0
    while step < total[0]:
        b = batches[step]
        x, m = collate([ids[i] for i in b], tok.pad_token_id)
        with torch.autocast('cuda', dtype=compute_dtype()):
            logits = net(input_ids=x, attention_mask=m).logits.squeeze(-1)
        loss = loss_fn(logits.float(), torch.from_numpy(y[b]).cuda())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step == 30 and args.minutes:
            per = (time.time() - t0) / 30
            t = torch.tensor([min(n_steps, int(args.minutes * 60 / per))], device='cuda')
            if world > 1:
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
            total[0] = int(t.item())
            log(rank, f'{per:.3f}s/step -> training {total[0]:,} of {n_steps:,} steps '
                      f'({total[0] * args.bs * world:,} pairs) in ~{total[0] * per / 60:.0f} min')
        if step % 50 == 0:
            el = time.time() - t0
            log(rank, f'step {step:,}/{total[0]:,}  loss {run:.4f}  lr {sched.get_last_lr()[0]:.2e}  '
                      f'{step * args.bs * world / el:,.0f} pairs/s  eta {(total[0] - step) * el / step / 60:.1f} min')
    if rank == 0:
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        log(rank, f'saved adapter to {args.out}')
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


@torch.inference_mode()
def score(args):
    """Score every pair in --data with a fine-tuned model; writes s1, s23 and logit_ce to --out."""
    rank, world = setup()
    tok = get_tok(args.model)
    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.iloc[:args.limit]
    idx = np.arange(rank, len(df), world)
    part = df.iloc[idx]
    if os.path.exists(os.path.join(args.adapter, 'adapter_config.json')):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model(args.model, tok), args.adapter).merge_and_unload()
    else:                                           # --full fine-tune: the saved dir is the whole model
        model = base_model(args.adapter, tok)
    model.cuda().eval()
    out = np.empty(len(part), np.float32)
    t0, done = time.time(), 0
    CHUNK = 250_000                                 # tokenise per chunk: one 2.5M-pair batch exhausts RAM
    for lo in range(0, len(part), CHUNK):
        ids = tokenize(tok, part.iloc[lo:lo + CHUNK], args.max_len)
        order = np.argsort([len(s) for s in ids], kind='stable')
        i = 0
        while i < len(order):                       # token-budget batches over length-sorted pairs
            n = max(1, args.tokens // len(ids[order[min(i + args.bs, len(order)) - 1]]))
            b = order[i:i + min(args.bs, n)]
            x, m = collate([ids[j] for j in b], tok.pad_token_id)
            with torch.autocast('cuda', dtype=compute_dtype()):
                out[lo + b] = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
            i += len(b)
        done += len(ids)
        el = time.time() - t0
        log(rank, f'{done:,}/{len(part):,} per rank  {done * world / el:,.0f} pairs/s  '
                  f'eta {(len(part) - done) * el / done / 60:.1f} min')
        del ids
    res = pd.DataFrame({'row': idx, 'logit_ce': out})
    res.to_parquet(f'{args.out}.rank{rank}')
    if world > 1:
        dist.barrier()
    if rank == 0:
        allr = pd.concat([pd.read_parquet(f'{args.out}.rank{r}') for r in range(world)]).sort_values('row')
        keep = [c for c in ('s1', 's23') if c in df.columns]
        res = df[keep].iloc[allr.row.to_numpy()].reset_index(drop=True).assign(logit_ce=allr.logit_ce.to_numpy())
        res.to_parquet(args.out)
        for r in range(world):
            os.remove(f'{args.out}.rank{r}')
        log(rank, f'wrote {len(res):,} scores to {args.out} in {(time.time() - t0) / 60:.1f} min')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'score'])
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--bs', type=int, default=64, help='train: pairs per GPU step; score: max pairs per batch')
    ap.add_argument('--tokens', type=int, default=48_000, help='score: max tokens per batch')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lora_r', type=int, default=32)
    ap.add_argument('--minutes', type=float, default=0, help='train: time budget (0 = one full epoch)')
    ap.add_argument('--gc', action='store_true', help='gradient checkpointing (24 GB GPUs with the 1.7B model)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full', action='store_true', help='train: full fine-tune instead of LoRA (small encoders)')
    a = ap.parse_args()
    if a.cmd == 'score':
        a.bs = max(a.bs, 1024)
    train(a) if a.cmd == 'train' else score(a)
