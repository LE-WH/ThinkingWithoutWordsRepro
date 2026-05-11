"""Eval Qwen3-4B (and Abstract-CoT variants) on MATH-500 with vLLM.

Two modes:
  --mode baseline      : standard chat template, thinking mode OFF (paper's baseline).
  --mode abstract      : prepend <beginabstract>, constrained-decode up to m_max
                         abstract tokens, then <endabstract>, then unconstrained answer.

Reports:
  - Accuracy via math_verify
  - Mean reasoning tokens, response tokens, total tokens
"""
import argparse, json, os, re, time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_verify import parse, verify


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def load_math500(path: str):
    rows = []
    with open(path) as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def build_baseline_prompts(tok: AutoTokenizer, rows):
    prompts = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["problem"]},
        ]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
    return prompts


def build_abstract_prompts(tok: AutoTokenizer, rows, begin_abs="<beginabstract>"):
    prompts = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["problem"]},
        ]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Inject begin-abstract immediately after the assistant prefix so the
        # model is forced into the abstract-trace state.
        prompts.append(text + begin_abs)
    return prompts


BOXED_RE = re.compile(r"\\boxed\{")


def extract_last_boxed(s: str):
    starts = [m.start() for m in BOXED_RE.finditer(s)]
    if not starts:
        return None
    i = starts[-1] + len("\\boxed{")
    depth = 1
    out = []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def is_correct(pred: str, gold: str) -> bool:
    if pred is None:
        return False
    try:
        gp = parse(f"${gold}$")
        pp = parse(f"${pred}$")
        return bool(verify(gp, pp))
    except Exception:
        return pred.strip() == gold.strip()


def count_split_tokens(tok, full_text: str, response_text: str):
    """Return (total_new_tokens, response_only_tokens). Reasoning = total - response."""
    n_total = len(tok(full_text, add_special_tokens=False).input_ids)
    n_resp = len(tok(response_text, add_special_tokens=False).input_ids)
    return n_total, n_resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", default="/workspace/data/math500.jsonl")
    ap.add_argument("--mode", choices=["baseline", "abstract"], default="baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 500")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--m-max", type=int, default=128, help="abstract trace length cap")
    ap.add_argument("--m-min", type=int, default=0, help="force at least N V_abs tokens before <endabstract> is allowed")
    ap.add_argument("--abs-temp", type=float, default=0.0, help="sampling temperature for abstract trace stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end-abstract-token", default="<endabstract>")
    ap.add_argument("--begin-abstract-token", default="<beginabstract>")
    args = ap.parse_args()

    rows = load_math500(args.data)
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.mode == "baseline":
        prompts = build_baseline_prompts(tok, rows)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop=None)
    else:
        prompts = build_abstract_prompts(tok, rows, begin_abs=args.begin_abstract_token)
        # Two-stage generation. Stage 1: constrained to abstract vocab. Stage 2: unconstrained.
        # We approximate by:
        #   - allowed_token_ids = V_abs ∪ {<endabstract>}, force <endabstract> after m_max
        #   - then generate the response unconstrained.
        # vLLM's `allowed_token_ids` in SamplingParams handles stage 1; we run a second call for stage 2.
        sp = None  # built below per stage

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=False,
    )

    t0 = time.time()

    if args.mode == "baseline":
        outs = llm.generate(prompts, sp)
        completions = [o.outputs[0].text for o in outs]
        records = []
        n_correct = 0
        total_resp_tokens = []
        for r, p, c in zip(rows, prompts, completions):
            box = extract_last_boxed(c)
            ok = is_correct(box, r["answer"])
            n_correct += int(ok)
            n_total, n_resp = count_split_tokens(tok, c, c)
            total_resp_tokens.append(n_total)
            records.append({
                "id": r.get("unique_id"),
                "problem": r["problem"],
                "gold": r["answer"],
                "pred_boxed": box,
                "completion": c,
                "n_total_tokens": n_total,
                "correct": ok,
            })
        acc = n_correct / len(rows)
        mean_tok = sum(total_resp_tokens) / len(total_resp_tokens)
        print(f"BASELINE acc={acc*100:.2f}  mean_tokens={mean_tok:.1f}  n={len(rows)} time={int(time.time()-t0)}s")

    else:
        # Stage 1 — constrained to abstract vocab + end-delimiter
        abs_tokens = []
        for i in range(64):
            if i < 26:
                abs_tokens.append(f"<TOKEN_{chr(ord('A')+i)}>")
            else:
                j = i - 26
                abs_tokens.append(f"<TOKEN_{chr(ord('A')+j//26)}{chr(ord('A')+j%26)}>")
        end_id = tok.convert_tokens_to_ids(args.end_abstract_token)
        abs_ids = tok.convert_tokens_to_ids(abs_tokens)
        allowed = list(set(abs_ids + [end_id]))
        # Two-stage to enforce m_min if requested:
        #   1a) up to m_min tokens constrained to V_abs (no end_id, no stop) — forced length
        #   1b) up to (m_max - m_min) tokens constrained to V_abs ∪ {end_id} with stop on end_id
        # Pass skip_special_tokens=False so abstract tokens appear in .text and round-trip.
        abstract_token_ids_per_prompt = [None] * len(prompts)
        if args.m_min > 0:
            sp1a = SamplingParams(
                temperature=args.abs_temp, top_p=1.0, seed=args.seed,
                max_tokens=args.m_min, min_tokens=args.m_min,
                allowed_token_ids=abs_ids,  # no end_id
                skip_special_tokens=False,
            )
            outs1a = llm.generate(prompts, sp1a)
            prefix_ids = [list(o.outputs[0].token_ids) for o in outs1a]
            prefix1 = [o.outputs[0].text for o in outs1a]
            prompts_after_min = [p + ab for p, ab in zip(prompts, prefix1)]
            sp1b = SamplingParams(
                temperature=args.abs_temp, top_p=1.0, seed=args.seed,
                max_tokens=max(1, args.m_max - args.m_min),
                allowed_token_ids=allowed,
                stop_token_ids=[end_id],
                skip_special_tokens=False,
            )
            outs1b = llm.generate(prompts_after_min, sp1b)
            tail_ids = [list(o.outputs[0].token_ids) for o in outs1b]
            tail1 = [o.outputs[0].text for o in outs1b]
            abstract_pieces = [a + b for a, b in zip(prefix1, tail1)]
            abstract_token_ids_per_prompt = [p + t for p, t in zip(prefix_ids, tail_ids)]
        else:
            sp1 = SamplingParams(
                temperature=args.abs_temp, top_p=1.0, seed=args.seed,
                max_tokens=args.m_max,
                allowed_token_ids=allowed,
                stop_token_ids=[end_id],
                skip_special_tokens=False,
            )
            outs1 = llm.generate(prompts, sp1)
            abstract_pieces = [o.outputs[0].text for o in outs1]
            abstract_token_ids_per_prompt = [list(o.outputs[0].token_ids) for o in outs1]
        # Ensure trailing end-abstract is present
        prompts2 = []
        for p, ab in zip(prompts, abstract_pieces):
            if not ab.endswith(args.end_abstract_token):
                ab = ab + args.end_abstract_token
            prompts2.append(p + ab + "\n")
        sp2 = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        outs2 = llm.generate(prompts2, sp2)
        answers = [o.outputs[0].text for o in outs2]

        n_correct = 0
        records = []
        abs_lens, resp_lens = [], []
        for r, ab, ans, ab_ids in zip(rows, abstract_pieces, answers, abstract_token_ids_per_prompt):
            box = extract_last_boxed(ans)
            ok = is_correct(box, r["answer"])
            n_correct += int(ok)
            # Count abstract via actual generated ids (excludes any end_id stop token)
            end_id_val = end_id
            n_abs = sum(1 for t in (ab_ids or []) if t != end_id_val)
            n_resp = len(tok(ans, add_special_tokens=False).input_ids)
            abs_lens.append(n_abs)
            resp_lens.append(n_resp)
            records.append({
                "id": r.get("unique_id"),
                "problem": r["problem"],
                "gold": r["answer"],
                "abstract": ab,
                "response": ans,
                "pred_boxed": box,
                "n_abstract_tokens": n_abs,
                "n_response_tokens": n_resp,
                "correct": ok,
            })
        acc = n_correct / len(rows)
        ma = sum(abs_lens)/len(abs_lens)
        mr = sum(resp_lens)/len(resp_lens)
        print(f"ABSTRACT acc={acc*100:.2f}  mean_abs={ma:.1f}  mean_resp={mr:.1f}  mean_total={ma+mr:.1f}  n={len(rows)} time={int(time.time()-t0)}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
