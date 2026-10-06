# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0

"""Generate ``src/unicode_data.rs``: the Unicode tables ``src/unicode.rs``
reads, from Python's ``unicodedata`` and ``str`` methods, so the Rust side
answers exactly what the Python code it mirrors (laya's ``lang.py``) sees.

Run with a Python whose ``unicodedata`` is Unicode 15.0.0 (Python 3.12, the
version laya's fixtures come from; the output is deterministic)::

    python3 scripts/gen_unicode.py > src/unicode_data.rs

Two HF ``tokenizers`` quirks are not in Python's tables and are listed here
by hand (``scripts/probe_hf_unicode.py`` measures them against the
``tokenizers`` wheel):

* its GPT-2 pre-tokenizer regex runs on Oniguruma, whose ``\\p{L}`` /
  ``\\p{N}`` are Unicode 16.0: the letters and numbers 16.0 added over
  15.0 are ``ONIG_*_EXTRA`` below;
* its NFC is ``unicode-normalization-alignments`` with Unicode 9.0 data:
  the combining marks assigned since have class 0 there (``CCC_AFTER_9``),
  and U+11938 (Dives Akuru, 13.0) neither decomposes nor composes.
"""

import sys
import unicodedata

UNICODE = "15.0.0"

ONIG_LETTER_EXTRA = [
    (0x1C89, 0x1C8A), (0xA7CB, 0xA7CD), (0xA7DA, 0xA7DC), (0x105C0, 0x105F3),
    (0x10D4A, 0x10D65), (0x10D6F, 0x10D85), (0x10EC2, 0x10EC4), (0x11380, 0x11389),
    (0x1138B, 0x1138B), (0x1138E, 0x1138E), (0x11390, 0x113B5), (0x113B7, 0x113B7),
    (0x113D1, 0x113D1), (0x113D3, 0x113D3), (0x11BC0, 0x11BE0), (0x13460, 0x143FA),
    (0x16100, 0x1611D), (0x16D40, 0x16D6C), (0x18CFF, 0x18CFF), (0x1E5D0, 0x1E5ED),
    (0x1E5F0, 0x1E5F0), (0x2EBF0, 0x2EE5D),
]  # fmt: skip
ONIG_NUMBER_EXTRA = [
    (0x10D40, 0x10D49), (0x116D0, 0x116E3), (0x11BF0, 0x11BF9), (0x16130, 0x16139),
    (0x16D70, 0x16D79), (0x1CCF0, 0x1CCF9), (0x1E5F1, 0x1E5FA),
]  # fmt: skip
CCC_AFTER_9 = [
    (0x07FD, 0x07FD), (0x0898, 0x089F), (0x08CA, 0x08D3), (0x09FE, 0x09FE), (0x0C3C, 0x0C3C),
    (0x0D3B, 0x0D3C), (0x0EBA, 0x0EBA), (0x1715, 0x1715), (0x1ABF, 0x1ACE), (0x1DF6, 0x1DFA),
    (0xA82C, 0xA82C), (0x10D24, 0x10D27), (0x10EAB, 0x10EAC), (0x10EFD, 0x10EFF),
    (0x10F46, 0x10F50), (0x10F82, 0x10F85), (0x11070, 0x11070), (0x1133B, 0x1133B),
    (0x1145E, 0x1145E), (0x11839, 0x1183A), (0x1193D, 0x1193E), (0x11943, 0x11943),
    (0x119E0, 0x119E0), (0x11A34, 0x11A34), (0x11A47, 0x11A47), (0x11A99, 0x11A99),
    (0x11D42, 0x11D42), (0x11D44, 0x11D45), (0x11D97, 0x11D97), (0x11F41, 0x11F42),
    (0x16FF0, 0x16FF1), (0x1E08F, 0x1E08F), (0x1E130, 0x1E136), (0x1E2AE, 0x1E2AE),
    (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF),
]  # fmt: skip
AFTER_9_COMPOSITE = 0x11938

CATEGORIES = [
    "Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps",
    "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co",
]  # fmt: skip
FLAGS = [
    ("ALPHA", str.isalpha),
    ("DECIMAL", str.isdecimal),
    ("DIGIT", str.isdigit),
    ("NUMERIC", str.isnumeric),
    ("SPACE", str.isspace),
    ("UPPER", str.isupper),
]


def sigma_flags(ch):
    """(cased, case_ignorable) as Python's final-sigma rule in ``str.lower``
    reads them (``_PyUnicode_IsCased`` / ``_PyUnicode_IsCaseIgnorable``;
    ``unicodedata`` exposes neither): ``A ch Σ`` ends in a final sigma iff
    ``ch`` is ignorable or cased, ``A Σ ch`` iff it is ignorable or not
    cased. Cased is only consulted for a non-ignorable character."""
    t1 = ("A" + ch + "\u03a3").lower()[-1] == "\u03c2"
    t2 = ("A\u03a3" + ch).lower()[1] == "\u03c2"
    ignorable = t1 and t2
    return t1 and not ignorable, ignorable


def ranges_of(pairs):
    """Consecutive code points with the same value -> (first, last, value)."""
    out = []
    for cp, v in pairs:
        if out and out[-1][1] == cp - 1 and out[-1][2] == v:
            out[-1][1] = cp
        else:
            out.append([cp, cp, v])
    return out


def main():
    if unicodedata.unidata_version != UNICODE:
        sys.exit(
            f"unicodedata is {unicodedata.unidata_version}; run with a Python whose is {UNICODE}"
        )
    names = [n for n, _ in FLAGS] + [
        "CASED",
        "CASE_IGNORABLE",
        "ONIG_LETTER",
        "ONIG_NUMBER",
    ]
    bit = {n: 1 << i for i, n in enumerate(names)}
    onig_l = {cp for lo, hi in ONIG_LETTER_EXTRA for cp in range(lo, hi + 1)}
    onig_n = {cp for lo, hi in ONIG_NUMBER_EXTRA for cp in range(lo, hi + 1)}
    props, ccc, decomp, compose, lower = [], [], [], [], []
    for cp in range(0x110000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        f = sum(bit[n] for n, fn in FLAGS if fn(ch))
        cased, ignorable = sigma_flags(ch) if cat != "Cs" else (False, False)
        f |= bit["CASED"] * cased | bit["CASE_IGNORABLE"] * ignorable
        if cat[0] == "L" or cp in onig_l:
            f |= bit["ONIG_LETTER"]
        if cat[0] == "N" or cp in onig_n:
            f |= bit["ONIG_NUMBER"]
        if cat != "Cn" or f:
            props.append((cp, (CATEGORIES.index(cat), f)))
        if unicodedata.combining(ch):
            ccc.append((cp, unicodedata.combining(ch)))
        d = unicodedata.decomposition(ch)
        if d and not d.startswith("<"):
            parts = [int(x, 16) for x in d.split()]
            decomp.append((cp, parts[0], parts[1] if len(parts) > 1 else 0))
            if len(parts) == 2 and unicodedata.normalize("NFC", ch) == ch:
                compose.append((parts[0], parts[1], cp))
        if cat != "Cs" and ch.lower() != ch:
            lo = ch.lower()
            if len(lo) == 1:
                lower.append((cp, ord(lo)))
            else:
                assert cp == 0x130 and lo == "i\u0307", (hex(cp), lo)
    compose.sort()

    w = sys.stdout.write
    # REUSE-IgnoreStart
    w("// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins\n")
    w("// SPDX-License-Identifier: Apache-2.0\n\n")
    # REUSE-IgnoreEnd
    w(
        f"// @generated by scripts/gen_unicode.py from Python {sys.version.split()[0]}'s unicodedata\n"
    )
    w(f"// (Unicode {UNICODE}). Do not edit.\n\n")
    w(f'pub const UNICODE_VERSION: &str = "{UNICODE}";\n\n')
    w("/// General categories, indexed by [`PROPS`]' category field.\n")
    w(
        f"pub const CATEGORIES: [&str; {len(CATEGORIES)}] = [{', '.join(repr(c).replace(chr(39), chr(34)) for c in CATEGORIES)}];\n\n"
    )
    for n in names:
        w(f"pub const {n}: u16 = {bit[n]};\n")
    w("\n/// `(first, last, category, flags)`: every code point not covered is\n")
    w("/// unassigned (`Cn`) with no flags.\n")
    w("pub static PROPS: &[(u32, u32, u8, u16)] = &[\n")
    for lo, hi, (c, f) in ranges_of(props):
        w(f"    (0x{lo:X}, 0x{hi:X}, {c}, {f}),\n")
    w("];\n\n/// `(first, last, canonical combining class)`.\n")
    w("pub static CCC: &[(u32, u32, u8)] = &[\n")
    for lo, hi, c in ranges_of(ccc):
        w(f"    (0x{lo:X}, 0x{hi:X}, {c}),\n")
    w("];\n\n/// `(code point, first, second or 0)`: one level of canonical\n")
    w("/// decomposition (Hangul syllables are algorithmic, not listed).\n")
    w("pub static DECOMP: &[(u32, u32, u32)] = &[\n")
    for cp, a, b in decomp:
        w(f"    (0x{cp:X}, 0x{a:X}, 0x{b:X}),\n")
    w("];\n\n/// `(first, second, primary composite)`, sorted: the canonical\n")
    w("/// compositions NFC performs (composition exclusions left out).\n")
    w("pub static COMPOSE: &[(u32, u32, u32)] = &[\n")
    for a, b, cp in compose:
        w(f"    (0x{a:X}, 0x{b:X}, 0x{cp:X}),\n")
    w("];\n\n/// `(code point, str.lower())` for every single-character lowering\n")
    w("/// (U+0130's two-character one is special-cased in code).\n")
    w("pub static LOWER: &[(u32, u32)] = &[\n")
    for cp, lo in lower:
        w(f"    (0x{cp:X}, 0x{lo:X}),\n")
    w("];\n\n/// Combining marks Unicode assigned after 9.0: class 0 in the NFC of\n")
    w("/// HF `tokenizers` (`unicode-normalization-alignments`, Unicode 9.0).\n")
    w("pub static CCC_AFTER_9: &[(u32, u32)] = &[\n")
    for lo, hi in CCC_AFTER_9:
        w(f"    (0x{lo:X}, 0x{hi:X}),\n")
    w(
        "];\n\n/// The one canonical composite assigned after 9.0 (see [`CCC_AFTER_9`]).\n"
    )
    w(f"pub const AFTER_9_COMPOSITE: u32 = 0x{AFTER_9_COMPOSITE:X};\n")


if __name__ == "__main__":
    main()
