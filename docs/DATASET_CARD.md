# Dataset card: synthetic population and support tickets

## Contents
- `data/population.jsonl`: 20,000 fictional people (name, age, town, region, occupation, sector, employer,
  employer type, optional club and club role, arrival year, street, e-mail, phone).
- `data/train.jsonl` (2,400 tickets) and `data/test.jsonl` (800 tickets): support requests to a fictional utility
  company. One ticket per author; train and test authors are disjoint. Test authors are drawn from the first 1,000
  people, so every register size used by the attacker (1k, 5k, 20k) contains all of them.
- `data/test_shifted.jsonl` (800 tickets, post-hoc addition): the same 800 test authors, each with a new ticket
  whose identity mentions and signature use 15 templates that never occur in training (`MENTIONS_SHIFTED`,
  `CLOSINGS_SHIFTED` in `src/spl/population.py`). The request sentences are shared with the other splits. It
  measures whether a detector trained on the training templates generalizes to new phrasings.
- Each ticket: text, author id, request category (billing / outage / moving / contract / complaint), exact character
  spans for every identity mention with its attribute and level (direct / quasi / context), spans of task facts
  (amounts, issue dates), and a flag for tickets typed in lowercase (20%).

## Generation
`src/spl/population.py`. Attribute frequencies follow Zipf-like distributions so that some towns, occupations and
employers are rare. A ticket contains 1–4 identity mentions drawn from 13 templates, a category-specific request
sentence, and a closing line (one of three, one of which is a signed name). The shifted set uses 13 other mention
templates and 2 other closings. Hashes of the generated files are in `data/manifest.json` (seed in
`configs/experiment.json`).

## Privacy and provenance
All people, places, organizations, e-mail addresses (`example.org`) and phone numbers (`+00`) are invented. Given
and family names are common names combined at random. No real person's data, and no data from the MedAnon/Bosch
challenge, was used or consulted.

## Limits
Template text is far more regular than real tickets; contextual clues come from a short fixed list; the attacker's
register is complete and error-free. These choices make effects measurable and comparable across strategies; they
also mean that absolute re-identification rates are not estimates of real-world risk.
