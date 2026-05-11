"""Merge a LoRA adapter + modules_to_save into a base model, save full safetensors."""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True,
    )
    # peft saves modules_to_save (embed_tokens/lm_head) as full weights; load them
    print(f"base vocab: {model.get_input_embeddings().weight.shape}")
    model.resize_token_embeddings(len(tok))
    print(f"resized to: {model.get_input_embeddings().weight.shape}")
    print(f"loading adapter from {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved merged model to {args.out}")


if __name__ == "__main__":
    main()
