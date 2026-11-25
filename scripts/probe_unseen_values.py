"""Post-hoc probe: do the learned detectors generalize to attribute *values* never seen in training?

The shifted test set of run 2 changes the phrasing of identity mentions but keeps the closed vocabulary of names,
towns, occupations, employers, clubs and streets that the trained detector (S3) saw during training. This probe
builds a second fictional population whose values are all new (same structure, same sizes, same templates, same
generator), and applies the S3 detector trained in run 2 (no retraining), the off-the-shelf NER (S2) and the
non-learned strategies. The attacker uses the new vocabularies and a register of 5,000 new people.

Together with run 2 this gives a 2 x 2 design: phrasing seen / unseen x values seen / unseen.

Usage: python scripts/probe_unseen_values.py [detector_dir]
  detector_dir: the S3 model saved by run 2 (SPL_SAVE_DETECTOR); if absent, S3 is retrained from data/train.jsonl
  with the run-2 configuration and seed (about 6 minutes on 4 CPU threads).
Writes results/probe_unseen_values/{RESULTS.md, results.json, per_ticket.jsonl, anonymized.jsonl}.
"""

import collections
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spl import population as P  # noqa: E402

# ---- new vocabularies: same structure and sizes as the originals, no shared values ----------------------------
NEW = {
    "FIRST": ["Alma", "Bruno", "Celia", "Dario", "Edda", "Felix", "Gina", "Hugo", "Ilse", "Jonas", "Katya", "Lorenzo",
              "Maren", "Nico", "Orla", "Paolo", "Rhea", "Stefan", "Thea", "Ugo", "Valentin", "Wilma", "Yannick",
              "Zelda", "Arne", "Britt", "Colm", "Delia", "Esme", "Florin", "Greta", "Henrik", "Iris", "Joel", "Kasia",
              "Lukas", "Mira", "Noel", "Odile", "Petra", "Raoul", "Selma", "Tobias", "Ulla", "Vito", "Willa",
              "Yvonne", "Zeno", "Agnes", "Bodil"],
    "LAST": ["Achterberg", "Bramley", "Castell", "Dorsey", "Engel", "Fairbairn", "Grohl", "Hendry", "Ingram",
             "Jessop", "Kowal", "Larsen", "Merrick", "Nordin", "Olsen", "Penrose", "Quayle", "Rourke", "Sandoval",
             "Tarrant", "Upton", "Voss", "Winslow", "Yeager", "Zamora", "Aldous", "Birch", "Cormack", "Devlin",
             "Eriksen", "Farley", "Gilmore", "Haines", "Ibbotson", "Jarvis", "Kemp", "Lund", "Mercer", "Novak",
             "Orme", "Prescott", "Radley", "Sorensen", "Tate", "Vickers", "Wilder", "Yardley", "Zimmer", "Blythe",
             "Carver"],
    "REGIONS": {"North": ["Brackenford", "Cindermoor", "Dovecote", "Elderwell", "Fernhurst", "Gorsehill"],
                "South": ["Hartsmere", "Inkpen Bay", "Jessamy", "Kingsferry", "Larkspur", "Mossbank"],
                "East": ["Nettlecombe", "Orrin", "Pebbleford", "Quillan", "Rushmere", "Stonecrop"],
                "West": ["Tidewater", "Umberleigh", "Vinegar Hill", "Wrenfield", "Yewdale", "Zennor Cove"]},
    "SECTORS": {"health": ["midwife", "optician", "radiographer", "dietitian", "care worker"],
                "education": ["tutor", "curriculum adviser", "exam invigilator", "sports instructor",
                              "learning mentor"],
                "trades": ["bricklayer", "glazier", "plasterer", "locksmith", "stonemason"],
                "office": ["bookkeeper", "secretary", "claims handler", "tax adviser", "compliance officer"],
                "technology": ["systems administrator", "database engineer", "cloud architect", "QA tester",
                               "UX researcher"],
                "hospitality": ["waiter", "bartender", "pastry cook", "housekeeper", "event planner"],
                "transport": ["deckhand", "dispatcher", "van loader", "air traffic controller", "harbour master"],
                "public service": ["police officer", "lifeguard", "traffic warden", "refuse collector",
                                   "coastguard"]},
    "EMPLOYERS": {"hospital": ["Cindermoor Infirmary", "Larkspur Medical Centre", "St. Winifred Hospital"],
                  "school": ["Fernhurst Grammar", "Mossbank Infant School", "Quillan High"],
                  "company": ["Rushmere Textiles", "Tidewater Shipping", "Orrin Robotics", "Gorsehill Dairies",
                              "Wrenfield Insurance", "Pebbleford Consulting"],
                  "public body": ["Yewdale District Council", "Northern Water Board", "Coastal Ambulance Trust"],
                  "small business": ["Hartsmere Florist", "Elderwell Bike Shop", "Stonecrop Deli",
                                     "Kingsferry Laundrette"]},
    "CLUBS": ["Dovecote Harriers", "Elderwell Bowls Club", "Inkpen Bay Surf Club", "Jessamy Book Circle",
              "Nettlecombe Orchestra", "Pebbleford Tennis Club", "Umberleigh Model Railway Society",
              "Vinegar Hill Quiz Team", "Zennor Cove Swimmers", "Hartsmere Photography Club",
              "Mossbank Gardening Club", "Rushmere Darts League"],
    "STREETS": ["Tallow Lane", "Cobbler's Row", "Wharf Street", "Lime Kiln Road", "Netherfield Walk", "Dyer's Close",
                "Gallows Hill", "Weaver Way", "Thistle Crescent", "Brewery Approach", "Lark Rise", "Foundry Mews",
                "Candle Street", "Harbour Steps", "Sheep Street", "Buttercross", "Old Forge Lane", "Mariner Court",
                "Paddock Row", "Cooper's Yard"],
}


def flat(v):
    return [x for xs in v.values() for x in xs] if isinstance(v, dict) else list(v)


for k, v in NEW.items():          # same structure and sizes as the original, and no value shared with it
    old = getattr(P, k)
    if isinstance(v, dict):
        assert old.keys() == v.keys() and all(len(old[g]) == len(v[g]) for g in old), k
    else:
        assert len(old) == len(v), k
    assert not ({x.lower() for x in flat(old)} & {x.lower() for x in flat(v)}), k
    setattr(P, k, v)

# Modules that read the vocabularies at import time are imported only after the patch.
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spl import anonymize as A  # noqa: E402
from spl.attack import LinkageAttacker  # noqa: E402
from spl.detector import SpanDetector  # noqa: E402

CFG = json.loads((ROOT / "configs/experiment.json").read_text())
RUN2 = json.loads((ROOT / "results/run2/results.json").read_text())
OUT = ROOT / "results/probe_unseen_values"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("SPL_THREADS", CFG["threads"])))
POOL, SEED = CFG["main_pool"], CFG["seed"] + 100

people = P.make_population(POOL, SEED)
prng = random.Random(SEED)
authors = prng.sample(people[:CFG["test_author_pool"]], CFG["n_test"])
srng = random.Random(SEED + 1)
TESTS = {"values_unseen": [P.make_ticket(p, prng, f"nv{i:04d}") for i, p in enumerate(authors)],
         "values_unseen_shifted": [P.make_ticket(p, srng, f"nvs{i:04d}", shifted=True) for i, p in enumerate(authors)]}

det_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "logs/s3_detector")
if Path(det_dir).exists():
    det, det_source = SpanDetector(det_dir, None, seed=CFG["seed"]), "run-2 model loaded from disk"
else:
    det = SpanDetector(CFG["detector_model"], RUN2["model_revisions"]["detector"], seed=CFG["seed"])
    det.train([json.loads(line) for line in open(ROOT / "data/train.jsonl", encoding="utf-8")],
              epochs=CFG["detector_epochs"])
    det_source = "retrained from data/train.jsonl with the run-2 configuration"
ner = A.NERStrategy(CFG["ner_model"], RUN2["model_revisions"]["ner"])
anon = {}
for split, rows in TESTS.items():
    anon[("S0_none", split)] = [A.s0_none(t) for t in rows]
    anon[("S1_regex", split)] = [A.s1_regex(t) for t in rows]
    anon[("S2_ner", split)] = ner.batch(rows)
    anon[("S3_trained_detector", split)] = [det.anonymize(t) for t in rows]
    anon[("S4_oracle_direct", split)] = [A.s4_oracle_direct(t) for t in rows]
    anon[("S5_oracle_generalize", split)] = [A.s5_oracle_generalize(t) for t in rows]
    anon[("S6_oracle_full", split)] = [A.s6_oracle_full(t) for t in rows]
STRATS = ["S0_none", "S1_regex", "S2_ner", "S3_trained_detector", "S4_oracle_direct", "S5_oracle_generalize",
          "S6_oracle_full"]

# ---- measurements (definitions identical to run_experiment.py) -------------------------------------------------
WORD = re.compile(r"[a-z0-9]+")


def survives(t, x, s):
    return t["text"][s["start"]:s["end"]].lower() in x.lower()


def other_word_retention(t, x):
    ident = [(s["start"], s["end"]) for s in t["spans"] if s["level"] != "task"]
    keep = "".join(ch if not any(a <= i < b for a, b in ident) else " " for i, ch in enumerate(t["text"]))
    words = collections.Counter(WORD.findall(keep.lower()))
    left = collections.Counter(WORD.findall(x.lower()))
    total = sum(words.values())
    return sum(min(c, left[w]) for w, c in words.items()) / total if total else np.nan


def ci(values, n=1000, seed=0):
    v = np.array([x for x in values if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    b = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return {"point": float(v.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": int(len(v))}


att = LinkageAttacker(people)
per_ticket, res = [], {}
for split, rows_t in TESTS.items():
    for k in STRATS:
        arng = np.random.default_rng(CFG["seed"])
        rows = []
        for t, x in zip(rows_t, anon[(k, split)]):
            r = {"split": split, "strategy": k, "tid": t["tid"], "lowercase": t["lowercase"],
                 f"link_top1_{POOL}": att.attack(x, t["author"], arng)["top1"],
                 "other_word_retention": other_word_retention(t, x)}
            for lvl in ("direct", "quasi", "context"):
                sp = [s for s in t["spans"] if s["level"] == lvl]
                r[f"leak_{lvl}"] = float(np.mean([survives(t, x, s) for s in sp])) if sp else np.nan
            for attr in ("name", "age", "town", "occupation", "employer", "club", "street"):
                sp = [s for s in t["spans"] if s["attr"] == attr]
                r[f"kept_{attr}"] = float(np.mean([survives(t, x, s) for s in sp])) if sp else np.nan
            rows.append(r)
        per_ticket += rows
        res[(split, k)] = {m: ci([r[m] for r in rows]) for m in rows[0] if m.startswith(("link", "leak", "other",
                                                                                         "kept"))}

out = {"detector": det_source, "seed": SEED, "register": POOL, "n_tickets": CFG["n_test"],
       "results": {f"{s}/{k}": v for (s, k), v in res.items()}}
(OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
with open(OUT / "per_ticket.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in per_ticket:
        f.write(json.dumps(r) + "\n")
with open(OUT / "anonymized.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for split, rows_t in TESTS.items():
        for k in STRATS:
            for t, x in zip(rows_t, anon[(k, split)]):
                f.write(json.dumps({"split": split, "strategy": k, "tid": t["tid"], "original": t["text"],
                                    "text": x}, ensure_ascii=False) + "\n")

fmt = lambda b: f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"  # noqa: E731
M = f"link_top1_{POOL}"
L = ["# Probe: attribute values never seen in training (post-hoc; generated by scripts/probe_unseen_values.py)", "",
     f"A second fictional population ({POOL:,} people, seed {SEED}) with entirely new names, towns, occupations, "
     "employers, clubs and streets, generated with the same code and templates. 800 test tickets with the original "
     "templates and 800 with the shifted templates, from the same authors. S3 is the detector trained in run 2 "
     f"({det_source}); no model is retrained on the new values. The attacker uses the new vocabularies and the new "
     f"{POOL:,}-person register. 95% bootstrap intervals over tickets.", "",
     "## Linkage top-1: phrasing x values (values seen = run 2)", "",
     "| Strategy | Phrasing seen, values seen | Phrasing unseen, values seen | Phrasing seen, values unseen | "
     "Phrasing unseen, values unseen |", "|---|---|---|---|---|"]
for k in STRATS:
    L.append(f"| {k} | {fmt(RUN2['strategies'][k][M])} | {fmt(RUN2['shifted'][k][M])} | "
             f"{fmt(res[('values_unseen', k)][M])} | {fmt(res[('values_unseen_shifted', k)][M])} |")
for split, title in (("values_unseen", "original phrasings"), ("values_unseen_shifted", "unseen phrasings")):
    L += ["", f"## Identity spans kept verbatim and other words kept, unseen values, {title}", "",
          "| Strategy | Direct | Quasi | Contextual | Other words kept |", "|---|---|---|---|---|"]
    for k in STRATS:
        r = res[(split, k)]
        L.append(f"| {k} | {fmt(r['leak_direct'])} | {fmt(r['leak_quasi'])} | {fmt(r['leak_context'])} | "
                 f"{fmt(r['other_word_retention'])} |")
L += ["", "## S3 and S2: share of spans kept by attribute, unseen values", "",
      "| Attribute | S2, original phrasings | S2, unseen phrasings | S3, original phrasings | S3, unseen phrasings |",
      "|---|---|---|---|---|"]
for attr in ("name", "age", "town", "occupation", "employer", "club", "street"):
    L.append(f"| {attr} | " + " | ".join(f"{res[(s, k)][f'kept_{attr}']['point']:.3f}"
                                         for k in ("S2_ner", "S3_trained_detector")
                                         for s in ("values_unseen", "values_unseen_shifted")) + " |")
L += ["", "## Linkage top-1 by casing, unseen values", "", "| Strategy | Split | Cased | Lowercase |", "|---|---|---|---|"]
for k in ("S2_ner", "S3_trained_detector"):
    for split in TESTS:
        rows = [r for r in per_ticket if r["split"] == split and r["strategy"] == k]
        L.append(f"| {k} | {split} | " + " | ".join(fmt(ci([r[M] for r in rows if r["lowercase"] == lc]))
                                                   for lc in (False, True)) + " |")
(OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
