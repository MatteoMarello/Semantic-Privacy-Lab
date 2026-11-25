import random

import numpy as np

from spl import anonymize as A
from spl.attack import LinkageAttacker, extract, person_values
from spl.population import make_population, make_ticket, make_tickets

PEOPLE = make_population(500, seed=3)


def test_spans_match_attribute_values():
    for t in make_tickets(PEOPLE, 200, seed=4):
        vals = {k: v.lower() for k, v in person_values(PEOPLE[t["author"]]).items()}
        for s in t["spans"]:
            surface = t["text"][s["start"]:s["end"]].lower()
            if s["level"] != "task":
                assert surface == vals[s["attr"]], (surface, s)


def test_oracles_remove_what_they_promise():
    for t in make_tickets(PEOPLE, 200, seed=5):
        out4, out6 = A.s4_oracle_direct(t).lower(), A.s6_oracle_full(t).lower()
        for s in t["spans"]:
            surface = t["text"][s["start"]:s["end"]].lower()
            if s["level"] == "direct":
                assert surface not in out4
            if s["level"] in ("direct", "quasi", "context") and s["attr"] != "age":
                assert surface not in out6
            if s["level"] == "task":
                assert surface in out6           # task facts are never touched by the oracles


def test_generalization_tags_are_readable_by_the_attacker():
    p = PEOPLE[0]
    found = set(extract(f"I am {A.generalize('age', str(p.age))} and live in {A.generalize('town', p.town)}."))
    assert ("age_band", str(p.age // 10 * 10)) in found and ("region", p.region.lower()) in found


def test_town_inside_employer_name_is_not_read_as_town():
    found = extract("I work at Ashby Town Council.")
    assert ("employer", "ashby town council") in found and ("town", "ashby") not in found


def test_full_disclosure_is_reidentified():
    att = LinkageAttacker(PEOPLE)
    p = PEOPLE[7]
    text = f"My name is {p.name}. You can reach me at {p.email}."
    assert att.attack(text, p.pid, np.random.default_rng(0))["top1"] == 1.0


def test_empty_text_gives_chance_level():
    att = LinkageAttacker(PEOPLE)
    r = att.attack("Hello, my bill is too high.", PEOPLE[3].pid, np.random.default_rng(0))
    assert r["anonymity_set"] == len(PEOPLE) and abs(r["top1"] - 1 / len(PEOPLE)) < 1e-12


def test_regex_masks_email_and_phone():
    t = make_ticket(PEOPLE[1], random.Random(0), "x")
    t["text"] = f"Write to {PEOPLE[1].email} or call {PEOPLE[1].phone}."
    out = A.s1_regex(t)
    assert PEOPLE[1].email not in out and PEOPLE[1].phone not in out


def test_shifted_tickets_have_correct_spans_and_are_readable():
    rng = random.Random(9)
    att = LinkageAttacker(PEOPLE)
    hits = 0
    for p in PEOPLE[:150]:
        t = make_ticket(p, rng, "s", shifted=True)
        vals = {k: v.lower() for k, v in person_values(p).items()}
        for s in t["spans"]:
            if s["level"] != "task":
                assert t["text"][s["start"]:s["end"]].lower() == vals[s["attr"]]
        extracted = dict(extract(t["text"]))
        for s in t["spans"]:
            if s["attr"] in ("age", "arrival_year", "name", "email"):
                assert extracted.get(s["attr"]) == vals[s["attr"]], (s["attr"], t["text"])
        hits += att.attack(t["text"], p.pid, np.random.default_rng(0))["top1"]
    assert hits / 150 > 0.5          # unprotected shifted tickets are mostly re-identifiable
