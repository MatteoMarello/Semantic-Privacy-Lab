"""Synthetic population and support tickets with exact identity-span annotations.

Everything here is fictional: names are random combinations of common given names and surnames, towns, streets,
employers and clubs are invented, e-mail addresses use the reserved `example.org` domain and phone numbers use the
fictional prefix +00.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

FIRST = ["Anna", "Ben", "Carla", "David", "Elif", "Farid", "Grace", "Hamza", "Ines", "Jakob", "Kira", "Luis",
         "Mara", "Nils", "Olga", "Pavel", "Quinn", "Rosa", "Samir", "Tara", "Umar", "Vera", "Wim", "Yara", "Zoe",
         "Adam", "Bea", "Cyril", "Dina", "Emil", "Fleur", "Goran", "Hedda", "Ivan", "Jana", "Karl", "Lea", "Milo",
         "Nora", "Otto", "Pia", "Rami", "Sara", "Tomas", "Una", "Viktor", "Wanda", "Xavier", "Yusuf", "Zara"]
LAST = ["Abbott", "Berger", "Calloway", "Drummond", "Eckert", "Falk", "Gallo", "Hartley", "Imhof", "Jansen",
        "Keller", "Lindqvist", "Mancini", "Nowell", "Ortega", "Pratt", "Quist", "Ramsey", "Strand", "Thorne",
        "Ullman", "Varga", "Whitlock", "Yates", "Zeller", "Arden", "Brandt", "Corbin", "Dahl", "Ellery", "Fontaine",
        "Grieve", "Holm", "Ivers", "Joyce", "Kerr", "Lowe", "Moss", "Nagy", "Oakes", "Pike", "Rees", "Sutter",
        "Tull", "Vance", "Wald", "Young", "Zorn", "Bishop", "Crane"]
REGIONS = {"North": ["Ashby", "Brookmere", "Coldridge", "Dunholt", "Elmwick", "Farrow"],
           "South": ["Galeby", "Hollin", "Ivybridge", "Juniper Bay", "Kestrel Point", "Lowmarsh"],
           "East": ["Marrow", "Netherby", "Oxley", "Pennick", "Quarry End", "Redfern"],
           "West": ["Saltash Cove", "Thornbury", "Upwell", "Valebrook", "Westerly", "Yarrow"]}
SECTORS = {"health": ["nurse", "pharmacist", "paramedic", "dentist", "physiotherapist"],
           "education": ["teacher", "librarian", "lecturer", "school counsellor", "teaching assistant"],
           "trades": ["electrician", "plumber", "carpenter", "welder", "roofer"],
           "office": ["accountant", "paralegal", "office manager", "payroll clerk", "auditor"],
           "technology": ["software developer", "network technician", "data analyst", "web designer",
                          "IT support specialist"],
           "hospitality": ["chef", "barista", "hotel receptionist", "baker", "sommelier"],
           "transport": ["bus driver", "train conductor", "courier", "forklift operator", "ferry pilot"],
           "public service": ["firefighter", "postal worker", "park ranger", "council planner", "court clerk"]}
EMPLOYERS = {"hospital": ["St. Aldric Hospital", "Brookmere General", "Harbourview Clinic"],
             "school": ["Coldridge Academy", "Juniper Bay Primary", "Redfern College"],
             "company": ["Netherby Logistics", "Pennick Software", "Oxley Foods", "Farrow Engineering",
                         "Thornbury Bank", "Upwell Analytics"],
             "public body": ["Ashby Town Council", "Southern Rail Authority", "County Fire Service"],
             "small business": ["Galeby Bakery", "Hollin Garage", "Quarry End Café", "Westerly Print Shop"]}
CLUBS = ["Ashby Rowing Club", "Brookmere Chess Society", "Elmwick Choir", "Galeby Cycling Club",
         "Kestrel Point Sailing Club", "Marrow Allotment Society", "Oxley Climbing Club", "Redfern Drama Group",
         "Thornbury Running Club", "Yarrow Birdwatchers", "Netherby Bridge Club", "Upwell Football Club"]
CLUB_ROLES = ["member", "coach", "treasurer", "chairperson"]
STREETS = ["Mill Lane", "Orchard Row", "Kiln Street", "Ferry Road", "Chapel Walk", "Tanner's Yard", "Beacon Hill",
           "Rope Walk", "Granary Close", "Willow Crescent", "Station Approach", "Copper Lane", "Abbey Mews",
           "Lantern Street", "Quay Side", "Heron Way", "Pump Lane", "Saddler Street", "Market Square", "Bell Court"]

CATEGORIES = ["billing", "outage", "moving", "contract", "complaint"]
REQUEST = {
    "billing": ["My last bill was €{amount}, which is far more than usual.",
                "I was charged €{amount} twice for the same month.",
                "Could you explain the €{amount} charge on my latest invoice?"],
    "outage": ["The power went out on {date} and only came back the next morning.",
               "We have had no electricity since {date}.",
               "There were three short outages on {date}, and my freezer was damaged."],
    "moving": ["I am moving out on {date} and need a final meter reading.",
               "Please transfer my contract to my new address from {date}.",
               "I will move house on {date}; how do I close the account?"],
    "contract": ["I would like to switch to the green tariff from {date}.",
                 "Please confirm the new monthly payment of €{amount}.",
                 "Can I reduce my monthly instalment to €{amount}?"],
    "complaint": ["Your engineer missed the appointment on {date} without notice.",
                  "I have called four times and nobody has fixed the problem.",
                  "I was promised a refund of €{amount} and never received it."],
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

# Identity mention templates. Each tuple: (level, attributes revealed, template). Placeholders in braces become
# annotated spans; literal text around them is not identifying on its own.
MENTIONS = [
    ("direct", ["name"], "My name is {name}."),
    ("direct", ["email"], "You can reach me at {email}."),
    ("direct", ["phone"], "My phone number is {phone}."),
    ("quasi", ["age"], "I am {age} years old."),
    ("quasi", ["age", "occupation"], "I'm a {age}-year-old {occupation}."),
    ("quasi", ["town"], "I live in {town}."),
    ("quasi", ["occupation"], "I work as a {occupation}."),
    ("quasi", ["employer"], "I work at {employer}."),
    ("quasi", ["occupation", "employer"], "I am a {occupation} at {employer}."),
    ("context", ["club", "club_role"], "As {club_role_article} {club_role} of the {club}, I am often out in the evenings."),
    ("context", ["arrival_year"], "I have been a customer since I moved here in {arrival_year}."),
    ("context", ["street", "occupation"], "I'm the only {occupation} on {street}, so the neighbours know me."),
    ("context", ["street"], "My house is the blue one at the end of {street}."),
]
CLOSINGS = [("direct", ["name"], "Best regards, {name}"), (None, [], "Thanks in advance."), (None, [], "Regards.")]


@dataclass
class Person:
    pid: int
    name: str
    age: int
    town: str
    region: str
    occupation: str
    sector: str
    employer: str
    employer_type: str
    club: str | None
    club_role: str | None
    arrival_year: int
    street: str
    email: str
    phone: str


def _zipf_choice(rng: random.Random, values: list, s: float = 1.1):
    weights = [1 / (i + 1) ** s for i in range(len(values))]
    return rng.choices(values, weights)[0]


def make_population(n: int, seed: int) -> list[Person]:
    rng = random.Random(seed)
    towns = [(t, r) for r, ts in REGIONS.items() for t in ts]
    rng.shuffle(towns)
    occs = [(o, s) for s, os in SECTORS.items() for o in os]
    rng.shuffle(occs)
    emps = [(e, t) for t, es in EMPLOYERS.items() for e in es]
    rng.shuffle(emps)
    people, used_emails = [], set()
    for pid in range(n):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        town, region = _zipf_choice(rng, towns, 0.9)
        occ, sector = _zipf_choice(rng, occs, 0.8)
        emp, etype = _zipf_choice(rng, emps, 0.7)
        club = rng.choice(CLUBS) if rng.random() < 0.35 else None
        role = (rng.choices(CLUB_ROLES, [0.85, 0.07, 0.04, 0.04])[0]) if club else None
        email = f"{first.lower()}.{last.lower()}{rng.randrange(10, 99)}@example.org"
        while email in used_emails:
            email = f"{first.lower()}.{last.lower()}{rng.randrange(100, 999)}@example.org"
        used_emails.add(email)
        people.append(Person(pid, f"{first} {last}", int(min(89, max(18, rng.gauss(46, 16)))), town, region, occ,
                             sector, emp, etype, club, role, rng.randrange(1975, 2026), rng.choice(STREETS), email,
                             f"+00 5{rng.randrange(10, 99)} {rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"))
    return people


def _fill(template: str, values: dict[str, str]) -> tuple[str, list[tuple[int, int, str]]]:
    """Format template, returning text and (start, end, attribute) spans for each placeholder."""
    out, spans, pos = [], [], 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out.append(template[pos:m.start()])
        start = sum(len(x) for x in out)
        val = values[m.group(1)]
        out.append(val)
        if m.group(1) not in ("club_role_article",):
            spans.append((start, start + len(val), m.group(1)))
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def make_ticket(person: Person, rng: random.Random, tid: str) -> dict:
    cat = rng.choice(CATEGORIES)
    values = {k: str(v) for k, v in asdict(person).items() if v is not None}
    values["club_role_article"] = "an" if (person.club_role or "")[:1] in "aeiou" else "a"
    pool = [m for m in MENTIONS if all(a in values for a in m[1])]
    k = rng.choice([1, 2, 2, 3, 3, 4])
    mentions, seen = [], set()
    for m in rng.sample(pool, len(pool)):          # avoid repeating the same attribute twice
        if len(mentions) < k and not (set(m[1]) & seen):
            mentions.append(m)
            seen |= set(m[1])
    body = rng.choice(REQUEST[cat])
    facts = {"amount": f"{rng.randrange(20, 900)}.{rng.randrange(0, 99):02d}",
             "date": f"{rng.randrange(1, 28)} {rng.choice(MONTHS)}"}
    closing = rng.choice(CLOSINGS)
    segments = [(None, [], "Hello,")] + mentions[: len(mentions) // 2 + 1] + [(None, [], body)] + \
        mentions[len(mentions) // 2 + 1:] + [closing]
    text, spans = "", []
    for level, attrs, tmpl in segments:
        piece, sp = _fill(tmpl, {**values, **facts})
        offset = len(text) + (1 if text else 0)
        text = (text + " " + piece) if text else piece
        for s, e, attr in sp:
            if attr in ("amount", "date"):
                spans.append({"start": s + offset, "end": e + offset, "attr": attr, "level": "task"})
            else:
                spans.append({"start": s + offset, "end": e + offset, "attr": attr, "level": level})
    lowercase = rng.random() < 0.2          # casually typed tickets: NER relies on capitalisation
    if lowercase:
        text = text.lower()
    return {"tid": tid, "author": person.pid, "category": cat, "text": text, "spans": spans, "lowercase": lowercase,
            "levels": sorted({s["level"] for s in spans if s["level"] != "task"})}


def make_tickets(people: list[Person], n_tickets: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    authors = rng.sample(people, min(n_tickets, len(people)))
    return [make_ticket(p, rng, f"t{i:05d}") for i, p in enumerate(authors)]
