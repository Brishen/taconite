# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0

"""Write ``tests/fixtures/pyre.jsonl``: Python's ``re`` answers (``fullmatch``,
``search``, ``match``, or a compile error) for patterns x texts x flags, for
``src/pyre.rs``. The first line lists the texts; every other line is a
pattern and flags with ``"error": true`` or ``"results"``, one digit a text:
fullmatch * 4 + search * 2 + match. Deterministic; run with CPython 3.12+::

    python3 scripts/gen_regex_fixtures.py
"""

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pyre.jsonl"

PATTERNS = [
    # what span validators look like
    r"[A-Z][a-z]+(?: [A-Z][a-z]+)*",
    r"\d+(?:\.\d+)?\s?(?:mg|ml|g)",
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?[MKB]?",
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}",
    r"\d{4}-\d{2}-\d{2}",
    r"\(?\d{3}\)?[-. ]?\d{3}[-.]\d{4}",
    r"https?://\S+",
    r"^[A-Z]{2,5}$",
    r"inc\.?|llc|corp\.?|ltd\.?",
    r"\b(?:the|a|an)\b",
    r"^(?!the\b)\w+",
    r"\w+(?<!ing)",
    r"(?<=\$)\d+",
    r"(?<!\d)\d{2}(?!\d)",
    r"\bdr\.? \w+",
    r"[^\W\d_]+",
    r"[^aeiou\s]+",
    r"(\w)\1",
    r"(?P<w>\w+) (?P=w)",
    r"(?i)straße",
    r"(?-i:ABC)def",
    r"(?x) \d+ # digits\n \s* (?: kg | lb )",
    r"a{2,3}",
    r"a{,2}b",
    r"a{2}",
    r"x{a",
    r"a{3,2}",
    r"a*?b",
    r"a+?",
    r"(?:ab)*c",
    r"(?:ab)++c",
    r"a*+a",
    r"(a|ab)(c|bcd)(d*)",
    r".+",
    r"(?s).+",
    r"^.*$",
    r"(?m)^\w+$",
    r"\A\w+\Z",
    r"[\u00e9\u00e8]t\u00e9",
    r"\x41\u0042\U00000043",
    r"[a\-z]+",
    r"[-a]+",
    r"[]a]+",
    r"[\d.]+",
    r"[\s\S]{3}",
    r"\D\W\S",
    r"[[:alpha:]]",
    r"(?a)\w+",
    r"(?a)\d",
    r"\d",
    r"\s",
    r"[\1]",
    r"(ab",
    r"ab)",
    r"*a",
    r"a**",
    r"\q",
    r"(?#comment)abc",
    r"[\b]",
    r"\bword\B",
    r"σ",
    r"[α-ω]+",
    r"\u0130",
    r"[k-m]",
    r"\u0131",
    r"µ",
    r"(?i)[^a-z]",
    r"",
    r"|a",
    r"(?:)",
    r"(a)|b",
    r"(?:a|b|c){2,}?d",
    r"(?=\w{5})\w+",
    r"(?<=ab|cd)e",
    r"(?<=a|bc)d",
    r"[😀-🙏]",
    r".",
    r"\n",
    r"$",
    r"a$",
    r"^$",
]

TEXTS = [
    "",
    "a",
    "ab",
    "aa",
    "aaa",
    "aab",
    "b",
    "abc",
    "ababc",
    "abcd",
    "abcbcd",
    "the",
    "The",
    "then",
    "Tim Cook",
    "tim cook",
    "Apple Inc.",
    "ACME",
    "400mg",
    "5 ml",
    "$2.5M",
    "$1,250",
    "March 15, 2024",
    "2024-03-15",
    "(555) 123-4567",
    "john.doe@example.com",
    "https://example.com/a?b=1",
    "running",
    "Dr. Smith",
    "hello hello",
    "STRASSE",
    "straße",
    "Straße",
    "ABCdef",
    "abcDEF",
    "12 kg",
    "12kg",
    "été",
    "ABC",
    "a-z",
    "]a]",
    "1.2.3",
    "x{a",
    "abe",
    "cde",
    "bcd",
    "ad",
    "😀",
    "line1\nline2",
    "end\n",
    "\n",
    "ΣΑΣ",
    "σς",
    "ΑΒΓ",
    "i",
    "I",
    "\u0131",
    "\u0130",
    "K",
    "\u212a",
    "\u017f",
    "S",
    "μ",
    "Μ",
    "\u0663\u0664",
    "a\u00a0b",
    "_x_",
    "word",
    "words",
]

FLAGS = [[], ["IGNORECASE"], ["MULTILINE"], ["IGNORECASE", "DOTALL"]]


def main():
    lines = [{"texts": TEXTS}]
    for pat in PATTERNS:
        for fl in FLAGS:
            f = 0
            for name in fl:
                f |= getattr(re, name)
            try:
                rx = re.compile(pat, f)
            except (re.error, OverflowError):
                lines.append({"pattern": pat, "flags": fl, "error": True})
                continue
            digits = "".join(
                str(
                    4 * (rx.fullmatch(t) is not None)
                    + 2 * (rx.search(t) is not None)
                    + (rx.match(t) is not None)
                )
                for t in TEXTS
            )
            lines.append({"pattern": pat, "flags": fl, "results": digits})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
