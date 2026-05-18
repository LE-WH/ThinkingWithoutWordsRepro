"""Build training sequences for Abstract-CoT warm-up.

Two phases:
  - "bottleneck": two-pass decomposition of the original block-attention bottleneck.
      Pass 1 — [X; C; Z̃] standard causal, loss on Z̃.   (Z̃ learns to compress C)
      Pass 2 — [X; Z̃; Y] standard causal, loss on Y.    (Y uses Z̃, never sees C)
      Both passes use standard causal masks → FlashAttention-2 compatible.
      Mathematically equivalent to the original single-pass with custom block mask.
  - "distill":    packed sequence [X; Z̃; Y] with standard causal mask.
                  Loss is computed on positions in (Z̃ ∪ Y).
"""
from __future__ import annotations
import json, random
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizer

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings


SYS = "Please reason step by step, and put your final answer within \\boxed{}."
IGNORE = -100


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def random_abstract_trace(M: int, abs_token_ids: List[int], m_min: int = 8, m_max: int = 64) -> List[int]:
    m = random.randint(m_min, m_max)
    return [random.choice(abs_token_ids) for _ in range(m)]


def encode_user_prefix(tok: PreTrainedTokenizer, user: str) -> List[int]:
    """Encode the chat-prefix up to (and including) the assistant header + the
    empty <think></think> block that Qwen3 inserts when enable_thinking=False.
    We will splice the abstract trace + answer afterward."""
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok(text, add_special_tokens=False).input_ids


def build_bottleneck_example(
    tok: PreTrainedTokenizer,
    user: str,
    cot: str,
    answer: str,
    abs_trace_ids: List[int],
    begin_id: int,
    end_id: int,
    max_len: int = 2048,
) -> Dict[str, Any] | None:
    """Return tensors for one packed bottleneck example.

    Layout (positions):
      [ X  |  C  | beginabs Z̃ endabs | Y eos ]
    Where: X = chat-prefix-with-empty-think,
           C = verbal CoT text (raw, no <think> tags since X already opened think),
           Z̃ section closes <think> automatically by ending with </think>? NO.
    For simplicity we treat the abstract trace as a *separate* segment inserted
    AFTER the empty <think></think> Qwen3 prefix, with explicit <beginabstract>...<endabstract>.
    The CoT segment is placed before the abstract trace so it can be bottlenecked-out.

    Mask:
      causal everywhere, plus rows in Y forbidden from cols in C.
    """
    X = encode_user_prefix(tok, user)
    C = tok(cot, add_special_tokens=False).input_ids
    Z = [begin_id] + abs_trace_ids + [end_id]
    # Newline before answer to mirror chat conventions
    Y = tok("\n" + answer, add_special_tokens=False).input_ids + [tok.eos_token_id]

    total = len(X) + len(C) + len(Z) + len(Y)
    if total > max_len:
        # Truncate C from the right to fit (keep prompt + abstract + answer intact).
        budget_for_C = max_len - (len(X) + len(Z) + len(Y))
        if budget_for_C < 16:
            return None
        C = C[:budget_for_C]
        total = len(X) + len(C) + len(Z) + len(Y)

    input_ids = X + C + Z + Y
    # Position ranges
    xs, xe = 0, len(X)
    cs, ce = xe, xe + len(C)
    zs, ze = ce, ce + len(Z)
    ys, ye = ze, total

    # Labels: ignore everywhere except Z̃ and Y (i.e., positions zs..ye-1 contribute to loss
    # when shifted: predict token at position p+1 from logits at position p).
    labels = [IGNORE] * total
    # We want loss on the *next-token prediction* of tokens in (Z̃ ∪ Y).
    # In standard causal LM, labels[i] = input_ids[i+1] and the loss at logit position i
    # corresponds to predicting input_ids[i+1]. We use the HF convention: labels has same
    # length as input_ids, with -100 to ignore, and the model shifts internally.
    for i in range(zs, ye):
        labels[i] = input_ids[i]

    # 4-D block attention mask: shape (T, T). 1 = attend, 0 = mask.
    T = total
    # Start from causal lower-triangular
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    # Forbid Y from attending to C: rows ys..ye-1, cols cs..ce-1 -> 0
    if len(C) > 0 and len(Y) > 0:
        mask[ys:ye, cs:ce] = False
    # Forbid Y from attending to the BEGIN_ABS/END_ABS delimiters? Paper says Y attends to Z (the
    # full abstract segment including delimiters). Keep delimiters attendable.

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask_2d": mask,  # we'll convert to additive in collator
        "lens": (len(X), len(C), len(Z), len(Y)),
    }


def build_distill_example(
    tok: PreTrainedTokenizer,
    user: str,
    answer: str,
    abs_trace_ids: List[int],
    begin_id: int,
    end_id: int,
    max_len: int = 2048,
) -> Dict[str, Any] | None:
    X = encode_user_prefix(tok, user)
    Z = [begin_id] + abs_trace_ids + [end_id]
    Y = tok("\n" + answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
    total = len(X) + len(Z) + len(Y)
    if total > max_len:
        # Truncate answer from the right to fit
        keep = max_len - len(X) - len(Z) - 1
        if keep < 16:
            return None
        Y = Y[:keep] + [tok.eos_token_id]
        total = len(X) + len(Z) + len(Y)
    input_ids = X + Z + Y
    xs, xe = 0, len(X)
    zs, ze = xe, xe + len(Z)
    ys, ye = ze, total
    labels = [IGNORE] * total
    for i in range(zs, ye):
        labels[i] = input_ids[i]
    # Standard causal mask only; collator will build from len.
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask_2d": None,
        "lens": (len(X), 0, len(Z), len(Y)),
    }


def build_bottleneck_pass1(
    tok: PreTrainedTokenizer,
    user: str,
    cot: str,
    abs_trace_ids: List[int],
    begin_id: int,
    end_id: int,
    max_len: int = 2048,
) -> Dict[str, Any] | None:
    """Pass 1 of two-pass bottleneck: [X; C; Z̃] with standard causal mask.

    Loss on Z̃ only. Z̃ attends to full X+C, learning to compress CoT.
    No custom mask → FlashAttention-2 compatible.
    """
    X = encode_user_prefix(tok, user)
    C = tok(cot, add_special_tokens=False).input_ids
    Z = [begin_id] + abs_trace_ids + [end_id]

    total = len(X) + len(C) + len(Z)
    if total > max_len:
        budget_for_C = max_len - len(X) - len(Z)
        if budget_for_C < 16:
            return None
        C = C[:budget_for_C]
        total = len(X) + len(C) + len(Z)

    input_ids = X + C + Z
    zs = len(X) + len(C)
    labels = [IGNORE] * zs + list(Z)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask_2d": None,
        "lens": (len(X), len(C), len(Z), 0),
    }


def build_bottleneck_pass2(
    tok: PreTrainedTokenizer,
    user: str,
    answer: str,
    abs_trace_ids: List[int],
    begin_id: int,
    end_id: int,
    max_len: int = 2048,
) -> Dict[str, Any] | None:
    """Pass 2 of two-pass bottleneck: [X; Z̃; Y] with standard causal mask.

    Loss on Y only. Y attends to X and Z̃ but never sees C.
    No custom mask → FlashAttention-2 compatible.
    """
    X = encode_user_prefix(tok, user)
    Z = [begin_id] + abs_trace_ids + [end_id]
    Y = tok("\n" + answer, add_special_tokens=False).input_ids + [tok.eos_token_id]

    total = len(X) + len(Z) + len(Y)
    if total > max_len:
        keep = max_len - len(X) - len(Z) - 1
        if keep < 16:
            return None
        Y = Y[:keep] + [tok.eos_token_id]
        total = len(X) + len(Z) + len(Y)

    input_ids = X + Z + Y
    ys = len(X) + len(Z)
    labels = [IGNORE] * ys + list(Y)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask_2d": None,
        "lens": (len(X), 0, len(Z), len(Y)),
    }


def collate(batch: List[Dict[str, Any]], pad_id: int):
    """Pad to longest in batch; build attention mask.

    When all examples use standard causal masking (attn_mask_2d is None), returns
    a 2-D padding mask [B, T] (1 = valid token, 0 = padding).  FA2 handles causal
    masking internally via is_causal; passing a 4-D additive mask causes FA2's
    _upad_input to misinterpret -inf entries as non-padding and OOM on 8k sequences.

    When any example has a custom block mask, falls back to a 4-D additive mask
    [B, 1, T, T] (0 = attend, -inf = block) for SDPA.
    """
    B = len(batch)
    T = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE, dtype=torch.long)
    has_custom = any(b["attn_mask_2d"] is not None for b in batch)

    if not has_custom:
        padding_mask = torch.zeros((B, T), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
            padding_mask[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": padding_mask}

    add_mask = torch.full((B, 1, T, T), float("-inf"), dtype=torch.bfloat16)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
        m = b["attn_mask_2d"]  # guaranteed not None in this branch
        m_f = torch.where(m, 0.0, float("-inf")).to(torch.bfloat16)
        add_mask[i, 0, :L, :L] = m_f
    return {"input_ids": input_ids, "labels": labels, "attention_mask": add_mask}


def collate_twopass(batch: List[Dict[str, Any]], pad_id: int):
    """Collator for two-pass bottleneck mode.

    Each item in batch is {"pass1": example, "pass2": example}.
    Returns (batch1, batch2) where each is a standard collated dict.
    """
    return (
        collate([item["pass1"] for item in batch], pad_id),
        collate([item["pass2"] for item in batch], pad_id),
    )
