"""Extend Qwen3-4B's tokenizer with the abstract vocab and delimiters, resize embeddings,
initialize the new rows, and save the result for downstream training.

Run:
  python code/extend_model.py --src Qwen/Qwen3-4B --out /workspace/runs/qwen3-4b-abs/base
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract import all_new_tokens, M_DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--init", choices=["mean", "zeros"], default="mean")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # extend on CPU, fast
    )
    pre = len(tok)
    new = all_new_tokens(args.M)
    n_added = tok.add_tokens(new, special_tokens=True)
    print(f"added {n_added} tokens (tokenizer: {pre} -> {len(tok)})")
    model.resize_token_embeddings(len(tok))

    embed = model.get_input_embeddings().weight.data
    lm_head = model.get_output_embeddings().weight.data
    new_ids = tok.convert_tokens_to_ids(new)
    assert all(i >= pre for i in new_ids), new_ids

    if args.init == "mean":
        mean_in = embed[:pre].mean(dim=0)
        mean_out = lm_head[:pre].mean(dim=0)
        for i in new_ids:
            embed[i] = mean_in + 1e-3 * torch.randn_like(mean_in)
            lm_head[i] = mean_out + 1e-3 * torch.randn_like(mean_out)
    elif args.init == "zeros":
        for i in new_ids:
            embed[i].zero_()
            lm_head[i].zero_()

    # Save
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "abstract_vocab.json"), "w") as f:
        json.dump({"new_tokens": new, "new_ids": new_ids, "M": args.M}, f, indent=2)
    print("saved to", args.out, "vocab=", len(tok))


if __name__ == "__main__":
    main()
