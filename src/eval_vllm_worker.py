"""Standalone vLLM eval worker — run as a subprocess by run_eval_vllm().

Separate script so vLLM's Engine Core subprocess re-imports THIS file (not the
training script), avoiding recursive initialisation.  The parent_process guard
prevents main() from running in vLLM's own worker subprocesses.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from multiprocessing import parent_process


def _extract_last_boxed(s: str):
    pat = re.compile(r"\\boxed\{")
    starts = [m.start() for m in pat.finditer(s)]
    if not starts:
        return None
    i, depth, out = starts[-1] + len("\\boxed{"), 1, []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1; out.append(c)
        elif c == "}":
            depth -= 1
            if depth > 0: out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True)
    ap.add_argument("--data",      required=True)
    ap.add_argument("--out",       required=True, help="JSON output path for results")
    ap.add_argument("--n",         type=int, required=True)
    ap.add_argument("--m-max",     type=int, required=True)
    ap.add_argument("--resp-max",  type=int, default=512)
    ap.add_argument("--gpu-util",  type=float, default=0.45)
    ap.add_argument("--tp",        type=int,   default=1, help="vLLM tensor parallel size (= number of training GPUs)")
    # Abstract-token IDs: optional — derived from tokenizer if omitted.
    ap.add_argument("--begin-id",  type=int,   default=None)
    ap.add_argument("--end-id",    type=int,   default=None)
    ap.add_argument("--abs-ids",   default=None, help="JSON list; derived from tokenizer if omitted")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_verify import parse, verify as _mv_verify

    def _check(pred, gold):
        if not pred:
            return False
        try:
            return bool(_mv_verify(parse(f"${gold}$"), parse(f"${pred}$")))
        except Exception:
            return (pred or "").strip() == gold.strip()

    SYS = "Please reason step by step, and put your final answer within \\boxed{}."

    # Load tokenizer early so we can derive abstract-token IDs if not provided.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.abs_ids is not None:
        abs_ids  = json.loads(args.abs_ids)
        begin_id = args.begin_id
        end_id   = args.end_id
    else:
        # Derive from the tokenizer (requires the model was built with extend_model.py)
        import sys as _sys
        _src = str(Path(__file__).parent)
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from abstract import abstract_token_strings, BEGIN_ABS, END_ABS
        abs_ids  = tok.convert_tokens_to_ids(abstract_token_strings(64))
        begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
        end_id   = tok.convert_tokens_to_ids(END_ABS)

    allowed   = set(abs_ids) | {end_id}

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[:args.n]

    nl_ids = tok.encode("\n", add_special_tokens=False)

    import torch as _torch
    tp = args.tp if args.tp >= 1 else _torch.cuda.device_count()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_util,
        dtype="bfloat16",
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Stage 1 — constrained abstract trace generation.
    # vLLM V1 dropped per-request logits_processors; use the built-in
    # allowed_token_ids which constructs a server-side mask instead.
    s1_prompts = []
    for r in rows:
        problem = r.get("problem", r.get("prompt", ""))
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": problem}]
        txt  = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids  = tok(txt, add_special_tokens=False).input_ids + [begin_id]
        s1_prompts.append({"prompt_token_ids": ids})

    s1_out = llm.generate(
        s1_prompts,
        SamplingParams(max_tokens=args.m_max + 2, temperature=0.0,
                       allowed_token_ids=sorted(allowed),
                       stop_token_ids=[end_id]),
        use_tqdm=False,
    )

    # Stage 2 — answer generation
    s2_prompts, traces = [], []
    for r, o1 in zip(rows, s1_out):
        z_ids = list(o1.outputs[0].token_ids)
        if not z_ids or z_ids[-1] != end_id:
            z_ids.append(end_id)
        z_clean = [t for t in z_ids if t not in (begin_id, end_id)]
        traces.append(z_clean)
        problem = r.get("problem", r.get("prompt", ""))
        msgs    = [{"role": "system", "content": SYS}, {"role": "user", "content": problem}]
        txt     = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        x_ids   = tok(txt, add_special_tokens=False).input_ids
        pfx     = x_ids + [begin_id] + z_ids + nl_ids
        s2_prompts.append({"prompt_token_ids": pfx})

    s2_out = llm.generate(
        s2_prompts,
        SamplingParams(max_tokens=args.resp_max, temperature=0.0,
                       stop_token_ids=[tok.eos_token_id]),
        use_tqdm=False,
    )

    # Score
    correct, abs_lens, resp_lens = 0, [], []
    for r, z_clean, o2 in zip(rows, traces, s2_out):
        gold = r["answer"]
        ans  = o2.outputs[0].text
        correct += int(_check(_extract_last_boxed(ans), gold))
        abs_lens.append(len(z_clean))
        resp_lens.append(len(o2.outputs[0].token_ids))

    n_rows = len(rows)
    abs_s  = sorted(abs_lens)
    results = {
        "acc":                  correct / n_rows,
        "mean_abstract_tokens": sum(abs_lens) / n_rows,
        "min_abstract_tokens":  abs_s[0] if abs_s else 0,
        "p25_abstract_tokens":  abs_s[n_rows // 4] if abs_s else 0,
        "p75_abstract_tokens":  abs_s[min(n_rows - 1, 3 * n_rows // 4)] if abs_s else 0,
        "max_abstract_tokens":  abs_s[-1] if abs_s else 0,
        "mean_response_tokens": sum(resp_lens) / n_rows,
        "mean_total_tokens":   (sum(abs_lens) + sum(resp_lens)) / n_rows,
        "n_eval":               n_rows,
    }
    with open(args.out, "w") as f:
        json.dump(results, f)


# Guard: vLLM's Engine Core subprocess re-imports __main__ via Python's spawn
# protocol.  parent_process() is not None in any subprocess, so main() only
# runs in the original worker process launched by run_eval_vllm().
if __name__ == "__main__" and parent_process() is None:
    main()
