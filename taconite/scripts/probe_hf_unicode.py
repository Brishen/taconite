# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0

"""Measure where HF ``tokenizers``' Unicode tables differ from Python's
``unicodedata``, for the lists ``gen_unicode.py`` carries by hand:

* the letters / numbers of the GPT-2 pre-tokenizer regex (Oniguruma's
  ``\\p{L}`` / ``\\p{N}``) that ``unicodedata`` does not call L* / N*;
* the characters whose NFC differs (``unicode-normalization-alignments``'
  tables are Unicode 9.0).

Every code point is probed through the ModernBERT tokenizer's own
pre-tokenizer and normalizer (a few seconds)::

    python3 scripts/probe_hf_unicode.py [laya checkpoint root]
"""

import sys
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer


def ranges(cps):
    out = []
    for cp in cps:
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return ", ".join("(0x%X, 0x%X)" % tuple(r) for r in out)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/bhawkins/laya")
    backend = AutoTokenizer.from_pretrained(str(root / "tokenizer")).backend_tokenizer
    pre, norm = backend.pre_tokenizer, backend.normalizer
    print("unicodedata", unicodedata.unidata_version)

    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp < 0xE000]
    # A letter continues `\p{L}+` after "a", a number `\p{N}+` after "1".
    joins = lambda head, c: len(pre.pre_tokenize_str(head + c)) == 1
    for cat, head in (("L", "a"), ("N", "1")):
        extra = [
            ord(c)
            for c in chars
            if unicodedata.category(c)[0] != cat and joins(head, c)
        ]
        missing = [
            ord(c)
            for c in chars
            if unicodedata.category(c)[0] == cat and not joins(head, c)
        ]
        print(f"\\p{{{cat}}} beyond unicodedata:", ranges(extra) or "none")
        print(f"\\p{{{cat}}} short of unicodedata:", ranges(missing) or "none")

    marks, decomps = [], []
    for c in chars:
        if unicodedata.combining(c):
            # U+0334 has class 1: NFC moves it before any mark of a higher class.
            s = "a" + c + "\u0334"
            if norm.normalize_str(s) != unicodedata.normalize("NFC", s):
                marks.append(ord(c))
        d = unicodedata.decomposition(c)
        if d and not d.startswith("<"):
            s = unicodedata.normalize("NFD", c)
            if norm.normalize_str(s) != unicodedata.normalize("NFC", s):
                decomps.append(ord(c))
    print("marks of class 0 to tokenizers:", ranges(marks) or "none")
    print("composites tokenizers does not compose:", ranges(decomps) or "none")


if __name__ == "__main__":
    main()
