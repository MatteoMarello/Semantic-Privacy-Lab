"""S3: a token classifier trained on the gold identity spans of the training tickets.

Labels: O / B-ID / I-ID, where ID covers direct, quasi and contextual identity spans (task facts such as amounts and
issue dates are O). Predicted spans are replaced by "[ID]"; regex masking of e-mails and phones is added on top.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .anonymize import regex_spans, replace_spans

LABELS = ["O", "B-ID", "I-ID"]


def encode(tok, ticket: dict, max_length: int = 192) -> dict:
    enc = tok(ticket["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
    ident = [(s["start"], s["end"]) for s in ticket["spans"] if s["level"] != "task"]
    labels = []
    for (a, b) in enc["offset_mapping"]:
        if a == b:
            labels.append(-100)
            continue
        lab = 0
        for s, e in ident:
            if a >= s and b <= e:
                lab = 1 if a == s or (labels and labels[-1] == 0) else 2
                break
        labels.append(lab)
    enc["labels"] = labels
    return enc


class SpanDetector:
    def __init__(self, name: str = "distilbert-base-cased", revision: str | None = None, seed: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForTokenClassification.from_pretrained(
            name, revision=revision, num_labels=3, id2label=dict(enumerate(LABELS)),
            label2id={lab: i for i, lab in enumerate(LABELS)})
        self.seed = seed

    def train(self, tickets: list[dict], epochs: int = 2, lr: float = 5e-5, batch_size: int = 16) -> list[dict]:
        data = [encode(self.tok, t) for t in tickets]
        steps = epochs * ((len(data) + batch_size - 1) // batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        rng, log, t0 = random.Random(self.seed), [], time.time()
        self.model.train()
        for ep in range(epochs):
            rng.shuffle(data)
            losses = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                n = max(len(x["input_ids"]) for x in batch)
                pad = self.tok.pad_token_id
                ids = torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in batch])
                att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in batch])
                lab = torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in batch])
                loss = self.model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
                losses.append(loss.item())
            log.append({"epoch": ep, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)})
        self.model.eval()
        return log

    @torch.no_grad()
    def predict_spans(self, ticket: dict) -> list[tuple[int, int]]:
        enc = self.tok(ticket["text"], truncation=True, max_length=192, return_offsets_mapping=True,
                       return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        pred = self.model(**enc).logits[0].argmax(-1).tolist()
        spans, cur = [], None
        for (a, b), p in zip(offsets, pred):
            if a == b:
                continue
            if p == 0:
                if cur:
                    spans.append(cur)
                    cur = None
            elif cur and (p == 2 or a <= cur[1] + 1):
                cur = (cur[0], b)
            else:
                if cur:
                    spans.append(cur)
                cur = (a, b)
        if cur:
            spans.append(cur)
        return spans

    def anonymize(self, ticket: dict) -> str:
        spans = [(s, e, "[ID]") for s, e in self.predict_spans(ticket)]
        return replace_spans(ticket["text"], spans + regex_spans(ticket["text"]))
