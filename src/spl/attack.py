"""Re-identification attacks against anonymized tickets.

A1 (attribute linkage): the attacker holds the population register and the public vocabularies (towns, regions,
occupations, sectors, employers, clubs, streets). It extracts every attribute value it can find in the text,
including generalized tags such as "[AGE 40-49]" or "[REGION North]", and ranks register records by the sum of the
IDF weights (log N / frequency) of the values they match.

A2 (dense retrieval): nearest register record in a sentence-embedding space.
"""

from __future__ import annotations

import collections
import math
import re

import numpy as np

from .population import CLUB_ROLES, CLUBS, EMPLOYERS, REGIONS, SECTORS, STREETS, Person

TOWNS = [t for ts in REGIONS.values() for t in ts]
OCCUPATIONS = [o for os in SECTORS.values() for o in os]
EMPLOYER_NAMES = [e for es in EMPLOYERS.values() for e in es]


def extract(text: str) -> list[tuple[str, str]]:
    """(attribute, value) pairs an attacker can read from the text. Values are lowercase."""
    t = text.lower()
    found = []
    # Longest multi-word entities first, blanked after matching so that e.g. the town inside
    # "Ashby Town Council" is not read as the author's town.
    for attr, vocab in (("employer", EMPLOYER_NAMES), ("club", CLUBS), ("street", STREETS),
                        ("occupation", OCCUPATIONS), ("town", TOWNS)):
        for v in sorted(vocab, key=len, reverse=True):
            vl = v.lower()
            if re.search(r"(?<![\w])" + re.escape(vl) + r"(?![\w])", t):
                found.append((attr, vl))
                t = t.replace(vl, " ")
    for m in re.finditer(r"\b([\w.+-]+@[\w-]+\.[\w.]+)\b", text.lower()):
        found.append(("email", m.group(1)))
    for m in re.finditer(r"(\+\d{2}(?:[ -]?\d){7,12})", text):
        found.append(("phone", m.group(1)))
    for m in re.finditer(r"\b(\d{2})(?: years old|-year-old)", t):
        found.append(("age", m.group(1)))
    for m in re.finditer(r"\[age (\d+)-\d+\]", t):
        found.append(("age_band", m.group(1)))
    for m in re.finditer(r"\[region (\w+)\]", t):
        found.append(("region", m.group(1)))
    for m in re.finditer(r"\[sector ([\w ]+?)\]", t):
        found.append(("sector", m.group(1)))
    for m in re.finditer(r"\[employer ([\w ]+?)\]", t):
        found.append(("employer_type", m.group(1)))
    for m in re.finditer(r"moved here in (\d{4})", t):
        found.append(("arrival_year", m.group(1)))
    for role in CLUB_ROLES:
        if re.search(rf"\b{role} of the\b", t):
            found.append(("club_role", role))
    for m in re.finditer(r"my name is ([a-z]+ [a-z]+)|regards, ([a-z]+ [a-z]+)", t):
        found.append(("name", m.group(1) or m.group(2)))
    return found


def person_values(p: Person) -> dict[str, str]:
    return {"name": p.name.lower(), "email": p.email.lower(), "phone": p.phone, "age": str(p.age),
            "age_band": str(p.age // 10 * 10), "town": p.town.lower(), "region": p.region.lower(),
            "occupation": p.occupation.lower(), "sector": p.sector.lower(), "employer": p.employer.lower(),
            "employer_type": p.employer_type.lower(), "club": (p.club or "").lower(),
            "club_role": p.club_role or "", "street": p.street.lower(), "arrival_year": str(p.arrival_year)}


class LinkageAttacker:
    def __init__(self, register: list[Person]):
        self.register = register
        self.n = len(register)
        self.index: dict[tuple[str, str], np.ndarray] = {}
        tmp = collections.defaultdict(list)
        for i, p in enumerate(register):
            for a, v in person_values(p).items():
                if v:
                    tmp[(a, v)].append(i)
        self.index = {k: np.array(v) for k, v in tmp.items()}
        self.pid = np.array([p.pid for p in register])

    def scores(self, text: str) -> np.ndarray:
        s = np.zeros(self.n)
        for key in set(extract(text)):
            idx = self.index.get(key)
            if idx is not None and len(idx):
                s[idx] += math.log(self.n / len(idx))
        return s

    def attack(self, text: str, author: int, rng: np.random.Generator) -> dict:
        s = self.scores(text)
        best = s.max()
        ties = np.flatnonzero(s == best)
        author_idx = int(np.flatnonzero(self.pid == author)[0])
        # rank with random tie-breaking
        order = np.lexsort((rng.random(self.n), -s))
        rank = int(np.flatnonzero(order == author_idx)[0]) + 1
        return {"top1": float(author_idx in ties) / len(ties), "top5": float(rank <= 5),
                "anonymity_set": int(len(ties)), "author_score_is_max": bool(author_idx in ties)}


def record_text(p: Person) -> str:
    club = f" {p.club_role} of the {p.club}." if p.club else ""
    return (f"{p.name}, {p.age} years old, {p.occupation} at {p.employer}, lives in {p.town} on {p.street}, "
            f"customer since {p.arrival_year}.{club} {p.email}, {p.phone}.")


class Embedder:
    """Mean-pooled sentence embeddings (sentence-transformers/all-MiniLM-L6-v2) with plain transformers."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).eval()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=256,
                               return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out)


def dense_attack(ticket_emb: np.ndarray, record_emb: np.ndarray, record_pids: np.ndarray,
                 authors: list[int]) -> list[dict]:
    sims = ticket_emb @ record_emb.T
    res = []
    for row, author in zip(sims, authors):
        top = np.argsort(-row)[:5]
        res.append({"top1": float(record_pids[top[0]] == author), "top5": float(author in record_pids[top])})
    return res
