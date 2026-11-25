# Semantic Privacy Lab

**How much does text anonymization actually protect, and what does it cost?** A controlled measurement of the
privacy–utility trade-off of seven anonymization strategies against two re-identification attacks, on a synthetic
population where the identity of every author and every identifying span is known exactly.

> **Status: work in progress.** Run 1 (the preregistered measurements) is complete:
> [`results/run1_preregistered_only/RESULTS.md`](results/run1_preregistered_only/RESULTS.md). Run 2, which adds a
> test set with unseen phrasings and an over-masking measure, has not been run yet.

## Why
Masking names and e-mail addresses is the default "anonymization" in many pipelines. But people are also identified
by combinations of ordinary facts ("a 34-year-old paramedic in Coldridge") and by context ("the only baker on Mill
Lane"). Real data cannot tell us how often a strategy fails, because the true author and the full set of
identifying spans are unknown. A synthetic population makes both known, so detection quality, residual
re-identification risk and utility loss can be measured for each strategy on identical inputs.

## Setup
- **Population**: 20,000 fictional people with skewed attribute frequencies (`src/spl/population.py`).
- **Tickets**: support requests to a fictional utility company, with a request category (utility label), task facts
  (amounts, dates) and 1–4 identity mentions at three levels: direct (name, e-mail, phone), quasi (age, town,
  occupation, employer), contextual (club role, arrival year, street, "the only … on …"). 20% are typed in lowercase.
  2,400 training and 800 test tickets from disjoint authors.
- **Strategies**: S0 none · S1 regex · S2 off-the-shelf NER (`dslim/bert-base-NER`) + regex · S3 DistilBERT span
  detector trained on the training tickets' gold spans + regex · S4 oracle masking of direct identifiers ·
  S5 S4 + generalization of quasi-identifiers (age band, region, sector, employer type) · S6 S5 + suppression of
  contextual clues. S4–S6 use gold spans: they are upper bounds on detection, not deployable systems.
- **Attacks**: A1 attribute linkage against the full population register (IDF-weighted matching of extracted
  values, including generalized tags); A2 dense retrieval with `all-MiniLM-L6-v2` embeddings of tickets and register
  records. Register sizes 1k / 5k / 20k.
- **Utility**: request-category macro-F1 (TF-IDF + logistic regression trained on equally anonymized text);
  share of task facts preserved; embedding similarity to the original text.
- **Uncertainty**: 95% bootstrap intervals over test tickets. Hypotheses were written down before running:
  [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).

## Results so far (run 1; 800 test tickets, register of 5,000 people, 95% bootstrap intervals)

| Strategy | Linkage re-identification (top-1) | Embedding similarity to original |
|---|---|---|
| S0 none | 0.863 [0.840, 0.883] | 1.000 |
| S1 regex | 0.712 [0.684, 0.740] | 0.985 [0.983, 0.987] |
| S2 off-the-shelf NER | 0.279 [0.254, 0.305] | 0.899 [0.893, 0.905] |
| S4 oracle, direct identifiers masked | 0.429 [0.402, 0.458] | 0.968 [0.966, 0.971] |
| S5 oracle, + quasi-identifiers generalized | 0.190 [0.170, 0.211] | 0.879 [0.874, 0.885] |
| S6 oracle, + contextual clues suppressed | 0.013 [0.010, 0.015] | 0.813 [0.806, 0.819] |

- Masking direct identifiers alone leaves 43% of authors re-identifiable from ordinary facts (H1 supported).
- Off-the-shelf NER is far weaker on lowercase tickets: 0.654 re-identification vs. 0.178 on normally cased ones.
- Risk falls with register size for every strategy (e.g. S4: 0.580 / 0.429 / 0.320 at 1k / 5k / 20k people; H4).
- Not reported as results: the trained span detector (S3) reaches 0.000 re-identification, but it was trained and
  tested on the same phrasings, so this measures memorization; run 2 tests it on unseen phrasings. The
  request-category F1 is 1.000 for every strategy (the category signal never overlaps identity spans), so the
  preregistered utility hypothesis H3 cannot be tested with that measure.

## Reproduce
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt
python -m pytest                           # generator, oracle and attacker tests
python scripts/run_experiment.py           # writes data/, results/ (CPU, about 1 hour)
```

## Relation to other work
This is an independent study. It does not reuse code, data or results from MedAnon, the Neural Wave hackathon
team project on clinical de-identification that I contributed to. The two share a theme: anonymization has to be
evaluated with attacks, not only with detection metrics.

## Limitations
Synthetic, template-based text; a fixed list of contextual clue types; an idealized attacker with a complete,
error-free register; English only. Absolute numbers are not estimates of real-world risk, and no strategy here
carries a formal privacy guarantee. What transfers is the comparison between strategies and the failure modes they
expose.

## License
MIT for code and the synthetic data. Models are downloaded from Hugging Face under their own licenses
(MIT / Apache-2.0).

AI assistance: code and documentation were drafted with Claude Code under my direction; design decisions and
results are documented so they can be checked independently.
