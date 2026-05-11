"""Shared definitions for Abstract-CoT: vocab, delimiters, constrained decoding."""
from __future__ import annotations
import torch
from transformers import LogitsProcessor


M_DEFAULT = 64
M_MAX_DEFAULT = 128
BEGIN_ABS = "<beginabstract>"
END_ABS = "<endabstract>"


def abstract_token_strings(M: int = M_DEFAULT):
    toks = []
    for i in range(M):
        if i < 26:
            toks.append(f"<TOKEN_{chr(ord('A')+i)}>")
        else:
            j = i - 26
            toks.append(f"<TOKEN_{chr(ord('A')+j//26)}{chr(ord('A')+j%26)}>")
    return toks


def all_new_tokens(M: int = M_DEFAULT):
    return [BEGIN_ABS, END_ABS] + abstract_token_strings(M)


class AbstractConstrainedLogits(LogitsProcessor):
    """Force the next m_max generated tokens to come from V_abs ∪ {END_ABS}, then end.

    After END_ABS is emitted (or m_max reached), this processor stops constraining.
    Designed to be applied *after* BEGIN_ABS has already been placed in the prompt.
    """

    def __init__(self, abs_token_ids, end_id: int, m_max: int, begin_id: int | None = None):
        self.allowed = set(int(t) for t in abs_token_ids) | {int(end_id)}
        self.allowed_tensor = torch.tensor(sorted(self.allowed), dtype=torch.long)
        self.end_id = int(end_id)
        self.begin_id = int(begin_id) if begin_id is not None else None
        self.m_max = int(m_max)

    def _count_abs_emitted(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Per batch row, count tokens since last BEGIN_ABS (or full seq if none seen).
        B = input_ids.shape[0]
        counts = torch.zeros(B, dtype=torch.long, device=input_ids.device)
        ends = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        for i in range(B):
            row = input_ids[i]
            # find latest BEGIN_ABS index, else 0
            if self.begin_id is not None:
                hits = (row == self.begin_id).nonzero(as_tuple=False)
                start = int(hits[-1].item()) + 1 if len(hits) > 0 else 0
            else:
                start = 0
            tail = row[start:]
            # If END_ABS already appeared, no further constraint
            if (tail == self.end_id).any():
                ends[i] = True
            counts[i] = tail.numel()
        return counts, ends

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # scores: (B, vocab)
        counts, already_ended = self._count_abs_emitted(input_ids)
        B, V = scores.shape
        mask = torch.full_like(scores, float("-inf"))
        allowed = self.allowed_tensor.to(scores.device)
        for i in range(B):
            if bool(already_ended[i]):
                mask[i] = scores[i]
                continue
            if int(counts[i]) >= self.m_max:
                # Force END_ABS
                row = torch.full((V,), float("-inf"), device=scores.device, dtype=scores.dtype)
                row[self.end_id] = 0.0
                mask[i] = row
            else:
                mask[i, allowed] = scores[i, allowed]
        return mask
