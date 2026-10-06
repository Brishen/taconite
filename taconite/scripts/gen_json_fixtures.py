# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0

"""Write ``tests/fixtures/json.jsonl``: Python's ``json`` round trips,
``repr(float)`` and ``round(x, n)``, for ``src/json.rs``. Deterministic.

Every line is one case, floats given by their IEEE bits (``bits``) so the
fixture does not depend on the parser under test:

    {"kind": "loads", "in": <text>, "out": <dumps>, "compact": <dumps with (",", ":")>}
    {"kind": "error", "in": <text json.loads rejects>}
    {"kind": "repr", "bits": <u64>, "out": <repr>}
    {"kind": "round", "bits": <u64>, "n": <ndigits>, "out_bits": <u64>}

    python3 scripts/gen_json_fixtures.py
"""

import json
import math
import random
import re
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "json.jsonl"


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


LOADS = [
    "null",
    "true",
    " false ",
    "0",
    "-0",
    "-0.0",
    "0.0",
    "1",
    "-1",
    "123456789012345678901234567890",
    "-98765432109876543210987654321",
    "18446744073709551616",
    "1e16",
    "1E16",
    "1e-5",
    "1e-4",
    "0.0001",
    "0.00001",
    "0.1",
    "0.3",
    "5e-324",
    "2.2250738585072014e-308",
    "1.7976931348623157e308",
    "1e400",
    "-1e400",
    "1.5e-7",
    "123456789012345680000.0",
    "9007199254740993",
    "9007199254740993.0",
    "1e15",
    "1e17",
    "1.0e+2",
    "100000.0",
    "12345.678",
    "NaN",
    "Infinity",
    "-Infinity",
    '""',
    '"plain"',
    '"quote \\" backslash \\\\ slash \\/ \\b\\f\\n\\r\\t"',
    '"\\u0000\\u0001\\u001f\\u007f\\u0080\\u00e9\\u2028\\u2029\\ufeff"',
    '"\\ud83d\\ude00 \\uD83D\\uDE00"',
    '"\\u00E9t\\u00e9"',
    '"raw é 日本 😀 \x7f"',
    "[]",
    "{}",
    '[1, 2.5, "x", null, true, [[]], {}]',
    '{"a": 1, "b": {"c": [1, {"d": "e"}]}, "é": "ü"}',
    '{"a": 1, "b": 2, "a": 3}',
    '{"z": 1, "a": 2, "z": {"n": 1}, "m": 3, "a": 4}',
    " \t\n\r[ 1 , 2 ]\n",
    '{"k":"v","n":[1,2,{"x":-0.5e-3}]}',
    "[1e5, 1E-5, 12e1, -12.5E+3, 0.5, 2e-7, 1e22, 1e21, 1e-7, 9.999999999999999e15]",
    '"\\ud800"',
    '"\\udc00x"',
    '"\\ud800\\u0041"',
]

ERRORS = [
    "",
    " ",
    "01",
    "1.",
    ".5",
    "+1",
    "1e",
    "[1,]",
    '{"a":1,}',
    "{'a': 1}",
    '"unterminated',
    '"bad \\x escape"',
    '"control \x01 char"',
    "nan",
    "infinity",
    "[1 2]",
    '{"a" 1}',
    "tru",
    "nul",
    "[",
    '{"a":',
    "1 2",
    '"\\u12"',
    '"\\u12G4"',
    "- 1",
    "--1",
]

REPRS = [
    0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, 1e16, 1e15, 9999999999999998.0, 1e17, 1e-4, 1e-5,
    0.00012345, 1.2345e-5, 1.5e-7, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 123456789.0,
    1234567890123456.7, 12345678901234567.0, 100000.0, 1e22, 1e100, 1e-100, 0.5, 0.25, 3.141592653589793,
    2.718281828459045, 1.1, 100.0, 1e5, 123.456, -123.456e-10, 4.35, 0.07, 1e21, 1.5e300,
]  # fmt: skip

ROUNDS = [
    (0.03125, 4), (0.03125, 3), (0.03125, 2), (0.00005, 4), (0.00015, 4), (0.00025, 4), (0.00035, 4),
    (0.00045, 4), (0.12345, 4), (0.12355, 4), (-0.00005, 4), (-0.03125, 4), (-0.00001, 4), (0.5, 0),
    (1.5, 0), (2.5, 0), (-2.5, 0), (2.675, 2), (1.005, 2), (0.125, 2), (0.375, 2), (1e-10, 4), (1e16, 4),
    (123456.78915, 4), (0.1 + 0.2, 4), (1 / 3, 4), (2 / 3, 4), (0.99995, 4), (0.99994999, 4),
    (0.99995000000001, 4), (1.7976931348623157e308, 4), (5e-324, 4), (-0.0, 4), (0.0, 4), (15.0, -1),
    (25.0, -1), (35.0, -1), (150.0, -2), (250.0, -2), (-250.0, -2), (12345.0, -3), (0.5, -1), (5.0, -1),
    (1234.5678, 20), (0.1, 17), (0.1, 30), (1e300, -300), (4.5e-5, 4), (0.95, 1), (0.85, 1), (0.45, 1),
]  # fmt: skip


def main():
    rng = random.Random(7)
    lines = []
    # The documented deviation: a lone surrogate becomes U+FFFD in Rust.
    lone = re.compile("[\ud800-\udfff]")
    for s in LOADS:
        v = json.loads(s)
        lines.append(
            {
                "kind": "loads",
                "in": s,
                "out": lone.sub("�", json.dumps(v, ensure_ascii=False)),
                "compact": lone.sub(
                    "�", json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                ),
            }
        )
    for s in ERRORS:
        try:
            json.loads(s)
        except (ValueError, RecursionError):
            lines.append({"kind": "error", "in": s})
        else:
            raise AssertionError(f"{s!r} parsed")
    reprs = list(REPRS)
    for _ in range(120):
        reprs.append(from_bits(rng.getrandbits(64)))
        reprs.append(rng.random() * 10 ** rng.randint(-30, 30))
        reprs.append(round(rng.random(), rng.randint(1, 8)))
    for x in reprs:
        if not math.isnan(x):
            lines.append({"kind": "repr", "bits": bits(x), "out": repr(x)})
    rounds = list(ROUNDS)
    for _ in range(200):
        k = rng.randint(0, 9999)
        rounds.append((k / 10000 + rng.choice([0, 0.00005, -0.00005]), 4))
        rounds.append((rng.uniform(-1, 1), rng.randint(0, 6)))
        rounds.append((k / 2 ** rng.randint(1, 20), rng.randint(0, 6)))
    for x, n in rounds:
        r = round(x, n)
        lines.append({"kind": "round", "bits": bits(x), "n": n, "out_bits": bits(r)})
    with open(OUT, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(len(lines), "cases")


if __name__ == "__main__":
    main()
