# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0

"""Write the fixtures ``tests/unicode.rs`` checks ``src/unicode.rs``
against, from Python (run it with the Python ``gen_unicode.py`` ran with):

* ``tests/fixtures/unicode_props.json``: for every code point, as ranges,
  ``str.isalpha`` / ``isdecimal`` / ``isdigit`` / ``isnumeric`` /
  ``isalnum`` / ``isspace`` / ``isupper``, ``unicodedata.category`` and
  ``unicodedata.combining``;
* ``tests/fixtures/unicode_cases.jsonl``: strings with their
  ``unicodedata.normalize("NFC", s)`` and ``s.lower()``.

    python3 scripts/gen_unicode_fixtures.py
"""

import json
import random
import sys
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
PREDICATES = [
    "isalpha",
    "isdecimal",
    "isdigit",
    "isnumeric",
    "isalnum",
    "isspace",
    "isupper",
]


def ranges(pred):
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        v = pred(chr(cp))
        if v:
            if out and out[-1][1] == cp - 1 and out[-1][2] == v:
                out[-1][1] = cp
            else:
                out.append([cp, cp, v])
    return out


def cases():
    rng = random.Random(99)
    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp < 0xE000]
    marks = [c for c in chars if unicodedata.combining(c)]
    decomposable = [
        c
        for c in chars
        if unicodedata.decomposition(c) and not unicodedata.decomposition(c)[0] == "<"
    ]
    lowerable = [c for c in chars if c.lower() != c]
    ignorable = [
        c for c in chars if unicodedata.category(c) in ("Mn", "Me", "Cf", "Lm", "Sk")
    ][:400]
    out = []
    for group in (decomposable, marks, lowerable):
        for i in range(0, len(group), 40):
            s = "".join(group[i : i + 40])
            out += [s, unicodedata.normalize("NFD", s)]
    for c in decomposable:
        out.append(unicodedata.normalize("NFD", c))
    for _ in range(1500):
        s = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.3:
                s.append(rng.choice(marks))
            elif r < 0.5:
                s.append(rng.choice(decomposable))
            elif r < 0.6:
                s.append(
                    chr(rng.randint(0x1100, 0x1112)) + chr(rng.randint(0x1161, 0x1175))
                )
                if rng.random() < 0.5:
                    s.append(chr(rng.randint(0x11A7, 0x11C3)))
            elif r < 0.7:
                s.append(chr(rng.randint(0xAC00, 0xD7A3)))
            elif r < 0.8:
                s.append(rng.choice("\u03a3\u03c3\u03c2\u0130I\u0131"))
            elif r < 0.9:
                s.append(rng.choice(ignorable))
            else:
                s.append(rng.choice(chars))
        out.append("".join(s))
    for ctx in (
        "",
        "A",
        "a",
        "1",
        " ",
        "'",
        "\u0345",
        "\u00b7",
        "\u02b0",
        "\u0301",
        "\u03a3",
        "\u01c5",
    ):
        for ctx2 in ("", "A", "a", "1", " ", ".", "\u0301", "\u02b0", "\u03a3"):
            out.append(ctx + "\u03a3" + ctx2)
            out.append("A" + ctx + "\u03a3" + ctx2 + "B")
    out += ["ΟΔΥΣΣΕΥΣ", "ΣΑΣ Σ ΑΣ.", "İstanbul İİ", "ǅǄǆ", "ẞ", "ΐ", "K", "Ω"]
    return out


def main():
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    props = {p: [r[:2] for r in ranges(getattr(str, p))] for p in PREDICATES}
    props["category"] = ranges(
        lambda c: unicodedata.category(c) if unicodedata.category(c) != "Cn" else None
    )
    props["combining"] = ranges(unicodedata.combining)
    props["unicode"] = unicodedata.unidata_version
    (OUT / "unicode_props.json").write_text(
        json.dumps(props, separators=(",", ":")) + "\n"
    )
    with open(OUT / "unicode_cases.jsonl", "w") as f:
        for s in cases():
            line = {"s": s, "nfc": unicodedata.normalize("NFC", s), "lower": s.lower()}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
