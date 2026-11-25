# Preregistration: Semantic Privacy Lab

The protocol is recorded in Git commit `f04d2b1`, before the implementation and archived Run 1 results in
the commit sequence. The early commits share a timestamp. The repository establishes their order and
contains no independently timestamped registration.

## Question
How much re-identification risk does each anonymization strategy remove, and how much utility does it cost, when
identity is revealed not only by direct identifiers but by combinations of quasi-identifiers and by contextual
clues? Everything is measured on a synthetic population where the ground truth (who wrote each text, which spans
identify them) is known exactly.

## Data (synthetic only)
- A fictional population (default 5,000 people): name, age, town, occupation, employer, club membership, year of
  arrival, e-mail, phone. Attribute frequencies are skewed, so some combinations are rare.
- Support tickets written by population members to a fictional utility company, generated from templates. Each
  ticket has a request category (the utility label: billing, outage, moving, contract, complaint), task-relevant
  facts (amounts, dates of the issue) and a random subset of identity mentions at three levels:
  direct (name, e-mail, phone), quasi (age, town, occupation, employer), contextual (club role, arrival year,
  descriptive phrases such as "the only <occupation> on <street>").
- Splits by author: train / test tickets have disjoint authors.

## Anonymization strategies
S0 none · S1 regex (e-mail, phone, numbers) · S2 off-the-shelf NER masking (`dslim/bert-base-NER`) + S1 ·
S3 trained span detector (DistilBERT token classifier fine-tuned on train tickets' gold identity spans) + S1 ·
S4 oracle masking of direct identifiers · S5 oracle masking of direct identifiers + generalization of quasi-identifiers
(age band, region, occupation sector, employer type) · S6 S5 + oracle suppression of contextual clues.
Oracle strategies are upper bounds for detection, not deployable systems.

## Attacks
- A1 attribute linkage: the attacker holds the population register (all attributes except the tickets), extracts
  attribute values and generalized values from the text with lexicons and patterns, and ranks candidates by
  IDF-weighted matches.
- A2 dense retrieval: `sentence-transformers/all-MiniLM-L6-v2` embeddings of the ticket vs. a textual description of
  each register record; nearest record.
- Metrics: top-1 and top-5 re-identification rate; median anonymity-set size (candidates tied at the best score).

## Utility
- U1 request-category macro-F1: TF-IDF + logistic regression trained and tested on text anonymized with the same
  strategy.
- U2 task-fact preservation: share of amounts and issue dates still present verbatim.
- U3 semantic similarity: cosine similarity between MiniLM embeddings of original and anonymized text.

## Hypotheses
- H1: masking direct identifiers alone (S4) leaves a top-1 re-identification rate above 20% under A1 (quasi and
  contextual clues remain).
- H2: off-the-shelf NER (S2) misses contextual clues and lowercase mentions; its re-identification rate is higher
  than S3 (trained detector).
- H3: generalization (S5, S6) reduces re-identification more than it reduces U1, whereas U2 is preserved by every
  strategy that does not touch numbers outside identity spans.
- H4: re-identification under A1 decreases with population size (1k → 5k → 20k) for every strategy.

## Analysis
95% bootstrap intervals over test tickets (1,000 resamples). Results by identity-mention level and by rarity of the
author's quasi-identifier combination.

## Scope
Synthetic text and a synthetic attacker. The numbers measure relative effects of strategies under controlled
conditions; they are not estimates of real-world re-identification risk and give no formal privacy guarantee.
