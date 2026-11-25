# Study guide: understanding and defending the methodology

This guide explains every design decision of the study, where it lives in the code, and the questions a careful
reviewer is likely to ask. Numbers are quoted from `results/`; the technical report is [`REPORT.md`](REPORT.md).

## 1. The question in one paragraph
Removing names and e-mail addresses from text is often called anonymization. People are also identified by
combinations of ordinary facts (age, town, occupation, employer: *quasi-identifiers*) and by context (club role,
street, "the only baker on Mill Lane"). The study measures, for seven anonymization strategies, how often an attacker
holding a population register can still find the author of a text (privacy), and how much of the text is lost
(utility). Synthetic data is used because it is the only way to know the true author and every identifying span
exactly.

## 2. The results in five sentences
1. Perfectly masking names, e-mails and phones (S4) still lets the attacker find 43% of authors: the policy, not
   detection, is the problem (H1).
2. Generalizing quasi-identifiers (S5) and deleting contextual clues (S6) lower this to 19% and 1.3%, but authors
   whose generalized attributes are rare stay more exposed.
3. Off-the-shelf NER (S2, 28%) beats S4 because it masks towns and employers too, but it never masks ages,
   occupations, arrival years or club roles, and largely fails on lowercase text (65% re-identified).
4. A detector trained on in-domain spans (S3) reaches ≈0% even on unseen phrasings and unseen values, but misses half
   of the ages in two unseen templates and starts deleting harmless words.
5. Every reduction in re-identification costs embedding similarity, in exactly the same order (ρ = 1.00); deletion
   of non-identity words only appears for S3 on unseen phrasings.

## 3. Pipeline, step by step

| Step | What happens | Code |
|---|---|---|
| Population | 20,000 fictional people; attribute values drawn from Zipf-like distributions, so some values are common and some rare | `src/spl/population.py: make_population` |
| Tickets | One support ticket per author: a greeting, 1–4 identity mentions from templates, a request sentence (carries the category and task facts), a closing. Every inserted value is recorded as a character span with its attribute and level. 20% of tickets are lowercased | `make_ticket`, `_fill` |
| Splits | 2,400 training and 800 test tickets with disjoint authors. Test authors come from the first 1,000 people so that every register (1k, 5k, 20k) contains them | `scripts/run_experiment.py` |
| Shifted test set (post-hoc) | The same 800 test authors write a second ticket whose identity mentions and closings use templates never seen in training | `MENTIONS_SHIFTED`, `CLOSINGS_SHIFTED` |
| Anonymization | S0–S6 applied to every split | `src/spl/anonymize.py`, `src/spl/detector.py` |
| Attacks | A1 linkage and A2 dense retrieval against a register of 1k / 5k / 20k people | `src/spl/attack.py` |
| Utility | U1 category classifier, U2 task facts kept, U3 embedding similarity, "other words kept" (post-hoc) | `scripts/run_experiment.py` |
| Statistics | Per-ticket outcomes → means with 95% bootstrap intervals | `ci`, `f1_ci` |
| Analyses | Hypothesis checks, rarity breakdown, generalization, residual risk, figures | `scripts/analyze.py` |

### Why the strategies were chosen
- **S0 none** and **S1 regex** (e-mail and phone patterns) are the floor: what happens with no or trivial effort.
- **S2 off-the-shelf NER** (`dslim/bert-base-NER`, masks PER/ORG/LOC/MISC) is the most common real-world approach.
- **S3 trained detector** (DistilBERT token classifier trained on the gold identity spans of the training tickets)
  is the best a learned system can do when it is trained for exactly this task.
- **S4–S6 oracles** use the gold spans. They are *not systems*: they answer "if detection were perfect, how much
  would this masking policy protect?" S4 masks direct identifiers only (the common policy), S5 also generalizes
  quasi-identifiers (age → band, town → region, occupation → sector, employer → type), S6 also deletes contextual
  clues. Comparing S4/S5/S6 separates the effect of the *policy* from the effect of *detection errors*.

### The attacker (threat model)
- **A1 attribute linkage** (`LinkageAttacker`). The attacker holds the full register (every attribute of every
  person) and the public vocabularies (towns, occupations, employers, clubs, streets, and the generalization tags).
  It reads every value it can find in the text (`extract`) and scores each register record by the sum of the IDF
  weights `log(N / count)` of the values it matches: a rare value counts more than a common one. This is a strong,
  idealized attacker: the register is complete and error-free, and the extraction patterns know how the templates
  phrase ages, arrival years, club roles and signatures, including the shifted phrasings. The shifted test set
  therefore tests the *defender's* generalization, not the attacker's.
- **A2 dense retrieval**. The ticket and a textual description of each register record are embedded with
  `all-MiniLM-L6-v2`; the nearest record is the guess. It needs no hand-written patterns but is much weaker
  (0.139 top-1 even on unprotected text), so it is a sanity check, not the main attack.

### Metrics
- **Linkage top-1** is an *expected* success rate: if the author is among `m` records tied at the best score, the
  attacker who picks one at random succeeds with probability `1/m`, so the ticket contributes `1/m`. A ticket with
  no readable clue contributes `1/N`.
- **Top-5**: author ranked in the first five after random tie-breaking.
- **Anonymity set**: number of records tied at the best score. A median of 1 means most authors are uniquely
  identified.
- **Span kept** (direct / quasi / contextual): share of gold spans whose surface string is still present verbatim
  (case-insensitive) in the anonymized text. This is a detection-recall measure. It slightly overcounts when the same
  characters occur elsewhere (an age of 31 inside the amount €231.40), which is why S5/S6 show a small non-zero
  "quasi kept"; the attacker is not affected by this artefact.
- **U1 category macro-F1**: TF-IDF + logistic regression trained on anonymized training tickets and tested on
  anonymized test tickets, both with the same strategy (the downstream task sees the anonymized text in both phases).
- **U2 task facts kept**: share of amounts and issue dates still present verbatim.
- **U3 embedding similarity**: cosine between MiniLM embeddings of the original and the anonymized ticket.
- **Other words kept** (post-hoc): share of the words *outside* identity spans still present (multiset count). It
  is 1.0 for a strategy that removes only identity spans and falls when a strategy deletes harmless words; it is the
  direct measure of over-masking. U3 cannot isolate over-masking, because removing identity spans also lowers the
  similarity.

### Statistics
- 95% percentile bootstrap intervals: tickets are resampled with replacement 1,000 times and the mean recomputed.
  They describe sampling uncertainty over *tickets* for a fixed population, fixed templates and fixed trained
  models; they do not cover variation over training seeds or over the template set.
- Differences between strategies, register sizes or test sets are computed *paired*: the same ticket (or the same
  author) appears in both arms, and the bootstrap resamples pairs. This removes between-author variation from the
  comparison.

## 4. What was preregistered and what was not

| Item | Status |
|---|---|
| Population, ticket design, strategies S0–S6, attacks A1/A2, metrics top-1/top-5/anonymity set, U1–U3, H1–H4, bootstrap, breakdown by mention level and by rarity | Recorded in the [research protocol](PREREGISTRATION.md) |
| Run 1 results | The preregistered measurements (`results/run1_preregistered_only/`, never modified) |
| Decision rules for H1–H4 | Written after run 1 (the preregistration states hypotheses, not thresholds); they are the literal reading of each sentence |
| Operational definition of "rarity" | Chosen after run 1 (generalized quasi-identifier class size in the 5k register) |
| Shifted-phrasing test set, "other words kept" | Post-hoc, motivated by two run-1 observations (S3 saturating at 0.000; U1 saturating at 1.000) |
| Attacker patterns for the shifted phrasings | Added together with the shifted set; run 2 reproduces the recorded attack outcomes exactly and the point estimates at the reported precision |
| Unseen-values probe (`scripts/probe_unseen_values.py`) | Post-hoc, exploratory |
| Template-level, attribute-level and residual-risk breakdowns, trade-off rank correlation | Post-hoc, exploratory |

Git records the protocol in `f04d2b1`, the implementation in `619e1f5` and the Run 1 archive with the
post-hoc additions in `f607d33`. The early commits share a timestamp, so they establish repository order.
An independently timestamped registration is unavailable.

Post-hoc analyses are legitimate but are hypothesis-*generating*: they were chosen after seeing data, so their
intervals should not be read as confirmatory tests.

## 5. Questions a reviewer will ask, with answers

**"Synthetic templates are not real text. Why should I believe anything here?"**
The absolute rates are not estimates of real-world risk and the report never claims they are. What the design
buys is exact ground truth, which real data cannot provide: we know the author and every identifying span, so we can
measure recall per identity level, residual risk, and over-masking without annotation error. The claims are
comparative (S4 vs S5, original vs unseen phrasings) and mechanistic (which clues re-identify).

**"The attacker is unrealistically strong."**
Yes, deliberately: a complete register and patterns that know the phrasings. A defence should be judged against a
strong attacker; a weaker attacker would make every strategy look safer. The dense attacker A2 is the weak,
pattern-free counterpart and is reported alongside.

**"S3 reaches 0.000 re-identification. Isn't that just memorization of the templates?"**
That was the concern after run 1, and it is why the shifted set exists. The post-hoc tests do not support it for
re-identification: on 15 unseen templates S3 stays at 0.003, and on a new population with unseen names, towns,
occupations, employers, clubs and streets it stays at 0.000 (original phrasings) and 0.005 (unseen phrasings).
What does degrade is detection in specific phrasings: about half of the ages in unseen templates survive (69% in
"At {age}, I did not expect this."), and S3 starts to delete harmless words (1.7%). Re-identification stays low
because an age alone is weak evidence (0.012 when it is the only readable clue). The honest summary is "generalizes
in aggregate, fails systematically on a few phrasings", and that the setting is favourable to S3 (same generator,
fixed clue types, short texts).

**"Then why bother with the oracles if S3 beats them?"**
S3 is compared with S6 under a different policy: S3 masks every identity span completely, S6 generalizes
quasi-identifiers (keeps "[AGE 30-39]"). S3 buys its lower re-identification with lower similarity (0.765 vs.
0.813). The oracles answer a question S3 cannot: what each *policy* achieves with perfect detection.

**"Why did you change the hypothesis about S3 after seeing the data?"**
No preregistered hypothesis was changed. H2 (S2 re-identifies more than S3) is evaluated as preregistered and
is supported. The value-memorization idea was a post-hoc explanation, tested with a post-hoc probe, and reported as
not supported.

**"Category macro-F1 is 1.000 for every strategy. Is the utility measure broken?"**
It is not broken, it is saturated: the category is fully determined by the request sentence, which never overlaps
an identity span, and no strategy removes words from it. That makes the preregistered H3 comparison uninformative,
which is stated as such rather than claimed as support. Over-masking is instead measured by "other words kept"
(post-hoc) and by embedding similarity.

**"Why report oracles at all?"**
They separate two causes of failure. If S4 (perfect masking of direct identifiers) still leaves 43% of authors
re-identifiable, the problem is the *policy*, and no better name detector can fix it. If S5/S6 work but S2/S3 do
not, the problem is *detection*.

**"Is S6 k-anonymous or differentially private?"**
Neither. S6 lowers measured re-identification under this attacker. It gives no formal guarantee; a different
attacker or a different clue type could do better. The rarity breakdown shows that authors in small generalized
classes remain more exposed.

**"Why does risk fall with register size?"**
With more people, each combination of values is shared by more of them, so the anonymity set grows and the
`1/m` success probability falls. It is the preregistered H4 and holds for every strategy that is not already at the
floor.

**"Only one training seed for S3?"**
Yes. The bootstrap intervals do not include seed variation. The in-distribution vs. shifted gap is large relative to
plausible seed variation, but this is a stated limitation.

**"Isn't the shifted set just one kind of shift?"**
Yes: it changes phrasing while keeping the vocabulary of values, the request sentences and the attacker fixed. It is
a lower bound on how much a template-trained detector can degrade, not a model of real distribution shift.

**"Multiple comparisons?"**
The preregistered hypotheses are four; the post-hoc tables are many and descriptive. No post-hoc difference is
presented as a confirmed finding.

## 6. Verify the numbers yourself
```bash
python -m pytest -q                       # generator, oracle and attacker tests
python scripts/analyze.py                 # recomputes every table in results/analysis/ from saved outputs (seconds)
python scripts/probe_unseen_values.py     # post-hoc probe; retrains S3 (~6 min) unless a saved model is given
SPL_THREADS=4 python scripts/run_experiment.py configs/experiment.json results/run2   # full re-run, ~15 min on 4 threads
```
Useful checks:
- `data/manifest.json` holds SHA-256 hashes of every data file; a re-run must reproduce them.
- `results/analysis/ANALYSIS.md`, section 1, compares run 2's original-test-set estimates with run 1 at the reported precision.
- `results/run2/anonymized.jsonl` contains every anonymized ticket, so any strategy's output can be read directly,
  e.g. `grep '"shift0001"' results/run2/anonymized.jsonl`.
- A single ticket can be attacked by hand:
  ```python
  import sys, json, numpy as np; sys.path.insert(0, "src")
  from spl.population import Person; from spl.attack import LinkageAttacker, extract
  people = [Person(**json.loads(l)) for l in open("data/population.jsonl")][:5000]
  text = "I'm the only baker on Mill Lane, so the neighbours know me."
  print(extract(text)); att = LinkageAttacker(people)
  print(np.sort(att.scores(text))[-5:])  # best scores; ties at the top form the anonymity set
  ```

## 7. Glossary
- **Direct identifier**: a value that identifies a person on its own (name, e-mail, phone).
- **Quasi-identifier**: a value shared by many people that identifies in combination (age, town, occupation,
  employer).
- **Contextual clue**: a description that is not a standard attribute field but narrows the candidates (club role,
  arrival year, street).
- **Generalization**: replacing a value with a coarser one (34 → 30–39).
- **Linkage attack**: matching values found in the text against an external register.
- **Anonymity set**: the candidates the attacker cannot distinguish from the author.
- **IDF weight**: `log(N / n_v)`, where `n_v` is the number of register members with value `v`.
- **Oracle**: a strategy given the ground truth, used as an upper bound.
- **Distribution shift**: test inputs that differ systematically from training inputs (here, phrasing).
- **Preregistration**: hypotheses and analyses written down before data are seen; protects against choosing
  analyses after seeing results.
- **Bootstrap**: estimating uncertainty by resampling the observed units with replacement.
