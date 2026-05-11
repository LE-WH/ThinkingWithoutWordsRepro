"""Generate on-policy abstract traces with vLLM.

Replaces the previous single-GPU HF `model.generate()` path. Uses vLLM's
`allowed_token_ids` SamplingParams field to enforce the V_abs ∪ {END_ABS}
alphabet directly in the sampler — no custom LogitsProcessor needed.

The prompt always ends with `<beginabstract>`, so the *first* generated token
onward is constrained to V_abs ∪ {END_ABS}. Generation terminates when
`<endabstract>` is sampled (via `stop_token_ids`) or when `m_max` tokens have
been produced (via `max_tokens`).
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings
from data_utils import encode_user_prefix, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Trained model dir")
    ap.add_argument("--data", default="/workspace/data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--use-cot", action="store_true",
                    help="If set, condition on x+c (for bottleneck t>=2). Else x only (distill).")
    ap.add_argument("--m-max", type=int, default=128)
    ap.add_argument("--max-prefix-len", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=2, help="vLLM tensor_parallel_size")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="vLLM max model length; must accommodate prefix + m_max")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # Tokenizer for prompt construction (vLLM owns its own copy internally).
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    abs_token_ids = tok.convert_tokens_to_ids(abstract_token_strings(64))
    begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
    end_id = tok.convert_tokens_to_ids(END_ABS)
    allowed_ids = [int(t) for t in abs_token_ids] + [int(end_id)]
    abs_set = set(int(x) for x in abs_token_ids)

    print(f"loading vLLM: base={args.base}  tp={args.tp}  max_model_len={args.max_model_len}")
    llm = LLM(
        model=args.base,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
    )

    rows = load_jsonl(args.data)[: args.n]
    print(f"building {len(rows)} prompts  use_cot={args.use_cot}")

    prompts = []
    for r in rows:
        X = encode_user_prefix(tok, r["prompt"])
        C = tok(r["cot"], add_special_tokens=False).input_ids if args.use_cot else []
        prefix = X + C + [begin_id]
        if len(prefix) > args.max_prefix_len:
            # Prefer truncating C from the right; if X alone overflows (long Dolci prompts),
            # keep the tail of X so the question stays intact.
            keep_for_C = args.max_prefix_len - len(X) - 1
            if keep_for_C >= 0:
                C = C[: keep_for_C]
            else:
                C = []
                X = X[-(args.max_prefix_len - 1):]
            prefix = X + C + [begin_id]
        prompts.append({"prompt_token_ids": prefix})

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.m_max,
        stop_token_ids=[int(end_id)],
        allowed_token_ids=allowed_ids,
        skip_special_tokens=False,
        detokenize=False,
        seed=args.seed,
    )

    t0 = time.time()
    outputs = llm.generate(prompts=prompts, sampling_params=sampling, use_tqdm=True)
    dt = int(time.time() - t0)

    # Parse outputs, write in original row order.
    out_traces = []
    for out in outputs:
        gen = list(out.outputs[0].token_ids)
        # stop_token_ids may include the stop token in output; strip it
        if end_id in gen:
            gen = gen[: gen.index(end_id)]
        trace = [t for t in gen if t in abs_set]
        if not trace:
            # Fallback: random short trace so downstream training never sees empty.
            trace = [random.choice(abs_token_ids) for _ in range(8)]
        out_traces.append(trace)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens = []
    with open(args.out, "w") as f:
        for trace in out_traces:
            f.write(json.dumps({"trace": trace}) + "\n")
            lens.append(len(trace))
    lens.sort()
    n = len(lens)
    rate = n / max(1, dt)
    print(f"DONE wrote {n} traces to {args.out}  t={dt}s  rate={rate:.1f}/s")
    print(f"  trace lens: mean={sum(lens)/n:.1f}  median={lens[n//2]}  "
          f"p10={lens[n//10]}  p90={lens[int(0.9*n)]}  max={max(lens)}")


if __name__ == "__main__":
    main()
