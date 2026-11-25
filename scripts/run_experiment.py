"""End-to-end Semantic Privacy Lab experiment. Writes data/, results/results.json, results/RESULTS.md, figures."""

import collections
import hashlib
import re
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from spl import anonymize as A  # noqa: E402
from spl.attack import Embedder, LinkageAttacker, dense_attack, record_text  # noqa: E402
from spl.detector import SpanDetector  # noqa: E402
from spl.population import make_population, make_ticket  # noqa: E402

CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.json"
CFG = json.loads(Path(CFG_PATH).read_text())
OUT_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
torch.set_num_threads(CFG["threads"])
OUT = Path("results")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.0f}s] {msg}", flush=True)


def rev(name):
    return HfApi().model_info(name).sha


# ---- data -------------------------------------------------------------------------------------
people = make_population(CFG["population"], CFG["seed"])
rng = np.random.default_rng(CFG["seed"])
import random  # noqa: E402

prng = random.Random(CFG["seed"])
test_authors = prng.sample(people[:CFG["test_author_pool"]], CFG["n_test"])
train_authors = prng.sample(people[CFG["test_author_pool"]:], CFG["n_train"])
test = [make_ticket(p, prng, f"test{i:04d}") for i, p in enumerate(test_authors)]
train = [make_ticket(p, prng, f"train{i:04d}") for i, p in enumerate(train_authors)]
# Distribution-shift test (post-hoc addition): same authors, phrasings never seen in training.
srng = random.Random(CFG["seed"] + 1)
test_shifted = [make_ticket(p, srng, f"shift{i:04d}", shifted=True) for i, p in enumerate(test_authors)]
TESTS = {"test": test, "test_shifted": test_shifted}
for name, rows in (("test", test), ("train", train), ("test_shifted", test_shifted)):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/population.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for p in people:
        f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
manifest = {k: hashlib.sha256(Path(f"data/{k}.jsonl").read_bytes()).hexdigest() for k in ("train", "test", "test_shifted", "population")}
Path("data/manifest.json").write_text(json.dumps({"config": CFG, "sha256": manifest}, indent=2))
log(f"data: {len(people)} people, {len(train)} train / {len(test)} test tickets")

# ---- strategies -------------------------------------------------------------------------------
models = {"ner": CFG["ner_model"], "detector": CFG["detector_model"], "embedder": CFG["embedder_model"]}
revisions = {k: rev(v) for k, v in models.items()}
anon = {"S0_none": {}, "S1_regex": {}, "S2_ner": {}, "S3_trained_detector": {}, "S4_oracle_direct": {},
        "S5_oracle_generalize": {}, "S6_oracle_full": {}}
simple = {"S0_none": A.s0_none, "S1_regex": A.s1_regex, "S4_oracle_direct": A.s4_oracle_direct,
          "S5_oracle_generalize": A.s5_oracle_generalize, "S6_oracle_full": A.s6_oracle_full}
ALL = {"train": train, **TESTS}
for split, rows in ALL.items():
    for k, fn in simple.items():
        anon[k][split] = [fn(t) for t in rows]
ner = A.NERStrategy(models["ner"], revisions["ner"])
for split, rows in ALL.items():
    anon["S2_ner"][split] = ner.batch(rows)
log("S2 NER done")
det = SpanDetector(models["detector"], revisions["detector"], seed=CFG["seed"])
det_log = det.train(train, epochs=CFG["detector_epochs"])
log(f"S3 detector trained {det_log}")
# S3 is trained on train tickets; its own training split is anonymized with the same model for the utility
# classifier (in-sample, as a deployed system would be applied to its own data).
for split, rows in ALL.items():
    anon["S3_trained_detector"][split] = [det.anonymize(t) for t in rows]
log("S3 applied")


# ---- per-ticket measurements ------------------------------------------------------------------
def survives(t, text_anon, span):
    return t["text"][span["start"]:span["end"]].lower() in text_anon.lower()


embedder = Embedder(models["embedder"], revisions["embedder"])
pools = {n: people[:n] for n in CFG["pool_sizes"]}
attackers = {n: LinkageAttacker(p) for n, p in pools.items()}
main_pool = CFG["main_pool"]
rec_emb = embedder.encode([record_text(p) for p in pools[main_pool]])
rec_pids = np.array([p.pid for p in pools[main_pool]])
log("embeddings ready")
WORD = re.compile(r"[a-z0-9]+")


def other_word_retention(t, x):
    """Share of words outside identity spans that are still present (multiset); measures over-masking."""
    ident = [(s["start"], s["end"]) for s in t["spans"] if s["level"] != "task"]
    keep = "".join(ch if not any(a <= i < b for a, b in ident) else " " for i, ch in enumerate(t["text"]))
    words = collections.Counter(WORD.findall(keep.lower()))
    left = collections.Counter(WORD.findall(x.lower()))
    total = sum(words.values())
    return sum(min(c, left[w]) for w, c in words.items()) / total if total else np.nan


def measure(split):
    rows_t = TESTS[split]
    orig_emb = embedder.encode([t["text"] for t in rows_t])
    out = {}
    for k in anon:
        texts = anon[k][split]
        arng = np.random.default_rng(CFG["seed"])
        rows = []
        emb = embedder.encode(texts)
        dense = dense_attack(emb, rec_emb, rec_pids, [t["author"] for t in rows_t])
        for i, (t, x) in enumerate(zip(rows_t, texts)):
            r = {"tid": t["tid"], "lowercase": t["lowercase"], "levels": t["levels"],
                 "dense_top1": dense[i]["top1"], "dense_top5": dense[i]["top5"],
                 "semantic_similarity": float(orig_emb[i] @ emb[i]),
                 "other_word_retention": other_word_retention(t, x)}
            for n, att in attackers.items():
                a = att.attack(x, t["author"], arng)
                r[f"link_top1_{n}"], r[f"link_top5_{n}"], r[f"anonset_{n}"] = a["top1"], a["top5"], a["anonymity_set"]
            task = [s for s in t["spans"] if s["level"] == "task"]
            r["task_fact_preserved"] = float(np.mean([survives(t, x, s) for s in task])) if task else np.nan
            for lvl in ("direct", "quasi", "context"):
                sp = [s for s in t["spans"] if s["level"] == lvl]
                r[f"leak_{lvl}"] = float(np.mean([survives(t, x, s) for s in sp])) if sp else np.nan
            rows.append(r)
        out[k] = rows
        log(f"{split} {k}: attacks done")
    return out


per_ticket_all = {split: measure(split) for split in TESTS}
per_ticket = per_ticket_all["test"]

# utility U1: category classifier trained and tested on text anonymized with the same strategy
u1 = {}
for k in anon:
    clf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = clf_vec.fit_transform(anon[k]["train"])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, [t["category"] for t in train])
    u1[k] = {split: list(clf.predict(clf_vec.transform(anon[k][split]))) for split in TESTS}
gold_cat = {split: [t["category"] for t in rows] for split, rows in TESTS.items()}


# ---- bootstrap ----------------------------------------------------------------------------------
def ci(values, n=1000, seed=0):
    v = np.array([x for x in values if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    b = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return {"point": float(v.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": int(len(v))}


def f1_ci(gold, pred, n=1000, seed=0):
    g, p = np.array(gold), np.array(pred)
    r = np.random.default_rng(seed)
    bs = [f1_score(g[i], p[i], average="macro") for i in (r.integers(0, len(g), len(g)) for _ in range(n))]
    return {"point": float(f1_score(g, p, average="macro")), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5))}


results = {"config": CFG, "model_revisions": revisions, "detector_training": det_log, "strategies": {}}
metric_keys = ([f"link_top1_{n}" for n in CFG["pool_sizes"]] + [f"link_top5_{main_pool}", "dense_top1", "dense_top5",
               "task_fact_preserved", "semantic_similarity", "other_word_retention", "leak_direct", "leak_quasi",
               "leak_context"])
results["shifted"] = {}
for k, rows in per_ticket_all["test_shifted"].items():
    sh = {m: ci([r[m] for r in rows]) for m in metric_keys}
    sh["category_macro_f1"] = f1_ci(gold_cat["test_shifted"], u1[k]["test_shifted"])
    results["shifted"][k] = sh
for k, rows in per_ticket.items():
    res = {m: ci([r[m] for r in rows]) for m in metric_keys}
    res["category_macro_f1"] = f1_ci(gold_cat["test"], u1[k]["test"])
    res[f"median_anonymity_set_{main_pool}"] = float(np.median([r[f"anonset_{main_pool}"] for r in rows]))
    res["by_lowercase"] = {str(lc): ci([r[f"link_top1_{main_pool}"] for r in rows if r["lowercase"] == lc])
                           for lc in (False, True)}
    res["by_levels_present"] = {}
    for combo in sorted({"+".join(r["levels"]) or "none" for r in rows}):
        sub = [r[f"link_top1_{main_pool}"] for r in rows if ("+".join(r["levels"]) or "none") == combo]
        res["by_levels_present"][combo] = {"n": len(sub), "top1": float(np.mean(sub))}
    results["strategies"][k] = res
results["seconds_total"] = round(time.time() - t_start, 1)
Path("results/results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
with open("results/per_ticket.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for split, pt in per_ticket_all.items():
        for k, rows in pt.items():
            for r in rows:
                f.write(json.dumps({"split": split, "strategy": k, **r}) + "\n")
examples = {k: {split: anon[k][split][:3] for split in TESTS} for k in anon}
Path("results/examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False), encoding="utf-8")

# ---- report -------------------------------------------------------------------------------------
fmt = lambda b: f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"  # noqa: E731
L = ["# Results (generated by scripts/run_experiment.py)", "",
     f"{len(test)} test tickets, register of {main_pool:,} people unless stated. 95% bootstrap intervals over tickets.",
     "", "## Privacy (lower is better) and utility (higher is better)", "",
     f"| Strategy | Linkage top-1 | Linkage top-5 | Dense top-1 | Median anonymity set | Category macro-F1 | "
     f"Task facts kept | Other words kept | Semantic similarity |", "|---|---|---|---|---|---|---|---|---|"]
for k, r in results["strategies"].items():
    L.append(f"| {k} | {fmt(r[f'link_top1_{main_pool}'])} | {fmt(r[f'link_top5_{main_pool}'])} | {fmt(r['dense_top1'])} | "
             f"{r[f'median_anonymity_set_{main_pool}']:.0f} | {fmt(r['category_macro_f1'])} | "
             f"{fmt(r['task_fact_preserved'])} | {fmt(r['other_word_retention'])} | {fmt(r['semantic_similarity'])} |")
L += ["", "## Identity spans that survive anonymization (share of gold spans still present verbatim)", "",
      "| Strategy | Direct | Quasi | Contextual |", "|---|---|---|---|"]
for k, r in results["strategies"].items():
    L.append(f"| {k} | {fmt(r['leak_direct'])} | {fmt(r['leak_quasi'])} | {fmt(r['leak_context'])} |")
L += ["", "## Linkage top-1 by register size", "", "| Strategy | " + " | ".join(f"{n:,}" for n in CFG["pool_sizes"]) + " |",
      "|---|" + "---|" * len(CFG["pool_sizes"])]
for k, r in results["strategies"].items():
    L.append(f"| {k} | " + " | ".join(fmt(r[f"link_top1_{n}"]) for n in CFG["pool_sizes"]) + " |")
L += ["", "## Linkage top-1 for lowercase vs. normally cased tickets", "", "| Strategy | Cased | Lowercase |", "|---|---|---|"]
for k, r in results["strategies"].items():
    L.append(f"| {k} | {fmt(r['by_lowercase']['False'])} | {fmt(r['by_lowercase']['True'])} |")
L += ["", "## Distribution shift: test tickets with phrasings never seen in training (post-hoc addition)", "",
      "| Strategy | Linkage top-1 | Direct kept | Quasi kept | Contextual kept | Other words kept |",
      "|---|---|---|---|---|---|"]
for k, r in results["shifted"].items():
    L.append(f"| {k} | {fmt(r[f'link_top1_{main_pool}'])} | {fmt(r['leak_direct'])} | {fmt(r['leak_quasi'])} | "
             f"{fmt(r['leak_context'])} | {fmt(r['other_word_retention'])} |")
Path("results/RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for k, r in results["strategies"].items():
    x, y = r[f"link_top1_{main_pool}"], r["semantic_similarity"]
    ax.errorbar(x["point"], y["point"], xerr=[[x["point"] - x["lo"]], [x["hi"] - x["point"]]],
                yerr=[[y["point"] - y["lo"]], [y["hi"] - y["point"]]], fmt="o", capsize=3)
    ax.annotate(k.split("_", 1)[0], (x["point"], y["point"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel(f"re-identification, linkage top-1 (register {main_pool:,})")
ax.set_ylabel("utility: embedding similarity to original")
ax.set_title("Privacy–utility trade-off (synthetic tickets)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/privacy_utility.png", dpi=150)
print("\n".join(L))
log("done")
