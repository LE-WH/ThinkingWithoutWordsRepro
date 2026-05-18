"""Post-training batch eval of saved LoRA checkpoints using vLLM TP=n.

Finds all *_step{N} adapter dirs produced by --save-every, merges each with
the base model, evaluates with vLLM, and prints a summary table.

Usage:
  python3 src/eval_checkpoints.py \
    --base  runs/qwen3-4b-abs/base \
    --out   runs/qwen3-4b-abs/pi1_phaseA \   # adapter dir (final) — also scans *_step* siblings
    --data  data/math500.jsonl \
    --n 500 --tp 4 \
    [--wandb-project PROJECT --wandb-run-id RUN_ID --phase phaseA]
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path


def merge_and_eval(base_dir: str, adapter_dir: str, data_path: str,
                   n: int, m_max: int, resp_max: int, tp: int) -> dict | None:
    worker = Path(__file__).parent / "eval_vllm_worker.py"
    tmp = tempfile.mkdtemp(prefix="eval_ckpt_")
    try:
        # Merge adapter into base on CPU.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("    merging...", end="", flush=True)
        t0 = time.time()
        tok   = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        del model
        import gc; gc.collect()
        print(f" {int(time.time()-t0)}s", flush=True)

        # Run vLLM eval (TP=tp, no DDP interference post-training).
        results_file = os.path.join(tmp, "results.json")
        proc = subprocess.run(
            [sys.executable, str(worker),
             "--model",    tmp,
             "--data",     data_path,
             "--out",      results_file,
             "--n",        str(n),
             "--m-max",    str(m_max),
             "--resp-max", str(resp_max),
             "--tp",       str(tp),
             "--gpu-util", "0.70",
            ],
            capture_output=False,
            timeout=3600,
        )
        if proc.returncode != 0:
            return None
        with open(results_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",  required=True, help="base model used for merging")
    ap.add_argument("--out",   required=True, help="final adapter dir (also scans *_step* siblings)")
    ap.add_argument("--data",  required=True)
    ap.add_argument("--n",        type=int, default=500)
    ap.add_argument("--m-max",    type=int, default=128)
    ap.add_argument("--resp-max", type=int, default=512)
    ap.add_argument("--tp",       type=int, default=4)
    ap.add_argument("--phase",    default="",  help="label for logging (e.g. phaseA)")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-run-id",  default="", help="resume existing run for logging")
    args = ap.parse_args()

    out_path = Path(args.out)

    # Collect: intermediate step checkpoints + final adapter
    step_dirs: list[tuple[int, Path]] = []
    parent = out_path.parent
    prefix = out_path.name
    for d in sorted(parent.glob(f"{prefix}_step*")):
        m = re.search(r"_step(\d+)$", d.name)
        if m and d.is_dir():
            step_dirs.append((int(m.group(1)), d))
    step_dirs.sort(key=lambda x: x[0])
    if out_path.exists():
        step_dirs.append((-1, out_path))   # -1 = final

    if not step_dirs:
        print(f"No checkpoints found (looked for {parent}/{prefix}_step* and {out_path})")
        return

    # W&B (resume existing training run to attach eval metrics)
    wb = None
    if args.wandb_project:
        try:
            import wandb as _wb
            init_kw = dict(project=args.wandb_project, resume="allow")
            if args.wandb_run_id:
                init_kw["id"] = args.wandb_run_id
            else:
                init_kw["name"] = f"eval_{args.phase or out_path.name}"
            wb = _wb.init(**init_kw)
        except Exception as e:
            print(f"W&B init failed: {e}")

    print(f"\n{'='*60}")
    print(f"Batch eval  base={args.base}  phase={args.phase or prefix}  n={args.n}  tp={args.tp}")
    print(f"{'='*60}")

    results: list[tuple[str, dict]] = []
    for step, d in step_dirs:
        label = "final" if step < 0 else f"step{step:05d}"
        print(f"\n[{label}] {d}")
        t_eval = time.time()
        metrics = merge_and_eval(args.base, str(d), args.data,
                                 args.n, args.m_max, args.resp_max, args.tp)
        elapsed = int(time.time() - t_eval)
        if metrics:
            print(f"    acc={metrics['acc']*100:.2f}%  "
                  f"abs={metrics['mean_abstract_tokens']:.1f} "
                  f"[p25={metrics.get('p25_abstract_tokens',0)} "
                  f"p75={metrics.get('p75_abstract_tokens',0)}]  "
                  f"resp={metrics['mean_response_tokens']:.1f}  t={elapsed}s")
            results.append((label, metrics))
            if wb is not None:
                log = {f"eval_{args.phase}/{k}": v for k, v in metrics.items()}
                wb.log(log, step=step if step >= 0 else None)
        else:
            print(f"    FAILED  t={elapsed}s")

    print(f"\n{'='*60}")
    print(f"Summary — {args.phase or prefix}")
    print(f"{'='*60}")
    for label, m in results:
        print(f"  {label:12s}  acc={m['acc']*100:.2f}%  "
              f"abs={m['mean_abstract_tokens']:.1f}  "
              f"resp={m['mean_response_tokens']:.1f}")

    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
