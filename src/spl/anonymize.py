"""Anonymization strategies. Each returns the anonymized text of a ticket.

Oracle strategies (S4–S6) use the gold spans and are upper bounds for detection quality, not deployable systems.
"""

from __future__ import annotations

import re

from .population import EMPLOYERS, REGIONS, SECTORS

TOWN_REGION = {t.lower(): r for r, ts in REGIONS.items() for t in ts}
OCC_SECTOR = {o.lower(): s for s, os in SECTORS.items() for o in os}
EMP_TYPE = {e.lower(): t for t, es in EMPLOYERS.items() for e in es}

DIRECT_TAG = {"name": "[NAME]", "email": "[EMAIL]", "phone": "[PHONE]"}
CONTEXT_ATTRS = {"club", "club_role", "arrival_year", "street"}
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"\+\d{2}(?:[ -]?\d){7,12}")


def replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace non-overlapping (start, end, replacement) spans, keeping the longest on overlap."""
    chosen, last_end = [], -1
    for s, e, rep in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if s >= last_end:
            chosen.append((s, e, rep))
            last_end = e
    for s, e, rep in reversed(chosen):
        text = text[:s] + rep + text[e:]
    return text


def generalize(attr: str, surface: str) -> str:
    v = surface.lower()
    if attr == "age":
        d = int(v) // 10 * 10
        return f"[AGE {d}-{d + 9}]"
    if attr == "town":
        return f"[REGION {TOWN_REGION[v]}]"
    if attr == "occupation":
        return f"[SECTOR {OCC_SECTOR[v]}]"
    if attr == "employer":
        return f"[EMPLOYER {EMP_TYPE[v]}]"
    raise KeyError(attr)


def s0_none(t: dict) -> str:
    return t["text"]


def regex_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), "[EMAIL]") for m in EMAIL_RE.finditer(text)] + \
           [(m.start(), m.end(), "[PHONE]") for m in PHONE_RE.finditer(text)]


def s1_regex(t: dict) -> str:
    return replace_spans(t["text"], regex_spans(t["text"]))


def s4_oracle_direct(t: dict) -> str:
    return replace_spans(t["text"], [(s["start"], s["end"], DIRECT_TAG[s["attr"]]) for s in t["spans"]
                                     if s["attr"] in DIRECT_TAG])


def s5_oracle_generalize(t: dict) -> str:
    spans = []
    for s in t["spans"]:
        surface = t["text"][s["start"]:s["end"]]
        if s["attr"] in DIRECT_TAG:
            spans.append((s["start"], s["end"], DIRECT_TAG[s["attr"]]))
        elif s["attr"] in ("age", "town", "occupation", "employer"):
            spans.append((s["start"], s["end"], generalize(s["attr"], surface)))
    return replace_spans(t["text"], spans)


def s6_oracle_full(t: dict) -> str:
    spans = []
    for s in t["spans"]:
        surface = t["text"][s["start"]:s["end"]]
        if s["attr"] in DIRECT_TAG:
            spans.append((s["start"], s["end"], DIRECT_TAG[s["attr"]]))
        elif s["attr"] in ("age", "town", "occupation", "employer"):
            spans.append((s["start"], s["end"], generalize(s["attr"], surface)))
        elif s["attr"] in CONTEXT_ATTRS:
            spans.append((s["start"], s["end"], "[REDACTED]"))
    return replace_spans(t["text"], spans)


class NERStrategy:
    """S2: off-the-shelf NER (PER/ORG/LOC/MISC masked) plus regex."""

    def __init__(self, model_name: str = "dslim/bert-base-NER", revision: str | None = None):
        from transformers import pipeline
        self.pipe = pipeline("token-classification", model=model_name, revision=revision,
                             aggregation_strategy="simple", device=-1)

    def batch(self, tickets: list[dict]) -> list[str]:
        outs = self.pipe([t["text"] for t in tickets], batch_size=16)
        res = []
        for t, ents in zip(tickets, outs):
            spans = [(int(e["start"]), int(e["end"]), f"[{e['entity_group']}]") for e in ents]
            res.append(replace_spans(t["text"], spans + regex_spans(t["text"])))
        return res
