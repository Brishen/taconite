// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5's tokenizer, bit-exact with HF `tokenizers`
//! (`tok(text, add_special_tokens=False)["input_ids"]`), std-only, read
//! straight from the checkpoint's `tokenizer.json` (the bundle carries a
//! copy).
//!
//! Encoding follows `TokenizerImpl::encode`:
//!
//! 1. Added tokens (`<|im_start|>`, `<think>`, ...) are cut out of the raw
//!    text: leftmost-longest matches.
//! 2. Every other piece is NFC-normalized (`tokenizers`' Unicode tables,
//!    [`unicode::nfc_hf`]).
//! 3. Pre-tokenization: the `Split` regex
//!    `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//!    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` (Oniguruma
//!    semantics, scanned by hand in [`qwen_split`]), then `ByteLevel` maps
//!    each byte to its GPT-2 symbol.
//! 4. BPE per piece, `Word::merge_all`'s queue replayed exactly (lowest
//!    rank first, leftmost on ties; the same code as taconite-laya's
//!    tokenizer).
//!
//! Decoding maps the GPT-2 symbols back to bytes (`ByteLevel` decoder).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use taconite::json::{self, Value};
use taconite::unicode;

/// The pre-tokenizer regex this tokenizer implements.
pub const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A multiplicative hash for the integer keys of the merge table.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (v ^ v >> 29).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }
}

type IdMap<K, V> = HashMap<K, V, BuildHasherDefault<IdHasher>>;

/// Added tokens: (content, id), by first byte, longest first.
#[derive(Default)]
struct AddedSet {
    tokens: Vec<(Vec<u8>, u32)>,
    by_first: Vec<Vec<usize>>,
}

impl AddedSet {
    fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let mut by_first = vec![Vec::new(); 256];
        for (i, t) in tokens.iter().enumerate() {
            by_first[t.0[0] as usize].push(i);
        }
        for v in &mut by_first {
            v.sort_by_key(|&i| Reverse(tokens[i].0.len()));
        }
        AddedSet { tokens, by_first }
    }

    /// `s` cut into pieces, each an added token (`Some(id)`) or text.
    fn split<'s>(&self, s: &'s str, out: &mut Vec<(&'s str, Option<u32>)>) {
        let b = s.as_bytes();
        let (mut done, mut i) = (0, 0);
        while i < b.len() {
            let hit = self.by_first[b[i] as usize].iter().find(|&&t| b[i..].starts_with(&self.tokens[t].0));
            let Some(&t) = hit else {
                i += 1;
                continue;
            };
            if done < i {
                out.push((&s[done..i], None));
            }
            let (content, id) = &self.tokens[t];
            out.push((&s[i..i + content.len()], Some(*id)));
            i += content.len();
            done = i;
        }
        if done < b.len() {
            out.push((&s[done..], None));
        }
    }
}

pub struct Tokenizer {
    /// Token strings (GPT-2 symbols): token `i` is `vocab[off[i]..off[i + 1]]`.
    vocab: Vec<u8>,
    off: Vec<u32>,
    /// Single-symbol tokens by symbol (GPT-2 symbols are below U+0200).
    small: Vec<u32>,
    /// (left, right) -> (rank, merged).
    merges: IdMap<u64, (u32, u32)>,
    added: AddedSet,
    /// added tokens by id: (content, special)
    added_by_id: HashMap<u32, (String, bool)>,
    token_ids: HashMap<String, u32>,
    sym: [char; 256],
    unsym: HashMap<char, u8>,
}

const NONE: u32 = u32::MAX;

/// GPT-2's `bytes_to_unicode`: the printable symbol each byte is spelled as.
fn byte_symbols() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).expect("Latin-1")
        } else {
            n += 1;
            char::from_u32(255 + n).expect("Latin Extended-A")
        };
    }
    table
}

impl Tokenizer {
    /// Read a Qwen3.5 `tokenizer.json`, refusing any configuration this
    /// implementation does not reproduce.
    pub fn from_hf_json(tokenizer_json: &str) -> Result<Tokenizer, String> {
        let spec = json::parse(tokenizer_json).map_err(|e| format!("tokenizer.json: {e}"))?;
        let bad = |what: &str| format!("tokenizer.json: unsupported {what}");
        let s = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_string);

        if s(spec.get("normalizer").and_then(|n| n.get("type"))).as_deref() != Some("NFC") {
            return Err(bad("normalizer (want NFC)"));
        }
        let pre = spec.get("pre_tokenizer").ok_or_else(|| bad("pre_tokenizer"))?;
        let steps = pre.get("pretokenizers").and_then(Value::as_array).ok_or_else(|| bad("pre_tokenizer"))?;
        let ok_split = steps.len() == 2
            && s(steps[0].get("type")).as_deref() == Some("Split")
            && s(steps[0].get("pattern").and_then(|p| p.get("Regex"))).as_deref() == Some(QWEN_PATTERN)
            && s(steps[0].get("behavior")).as_deref() == Some("Isolated")
            && steps[0].get("invert").and_then(Value::as_bool) == Some(false)
            && s(steps[1].get("type")).as_deref() == Some("ByteLevel")
            && steps[1].get("add_prefix_space").and_then(Value::as_bool) == Some(false)
            && steps[1].get("use_regex").and_then(Value::as_bool) == Some(false);
        if !ok_split {
            return Err(bad("pre_tokenizer (want Qwen's Split + ByteLevel)"));
        }
        let model = spec.get("model").ok_or_else(|| bad("model"))?;
        if s(model.get("type")).as_deref() != Some("BPE")
            || model.get("byte_fallback").and_then(Value::as_bool) == Some(true)
            || model.get("dropout").is_some_and(|d| !d.is_null())
        {
            return Err(bad("model (want byte-level BPE, no dropout)"));
        }

        let entries = model.get("vocab").and_then(Value::as_object).ok_or_else(|| bad("vocab"))?;
        let n = entries.len();
        let mut toks: Vec<Option<&str>> = vec![None; n];
        for (t, id) in entries {
            let id = id.as_i64().filter(|&i| (i as usize) < n).ok_or_else(|| bad("vocab id"))?;
            toks[id as usize] = Some(t);
        }
        let (mut vocab, mut off) = (Vec::new(), vec![0u32]);
        let mut token_ids = HashMap::with_capacity(n);
        for (i, t) in toks.iter().enumerate() {
            let t = t.ok_or_else(|| bad("vocab (ids not dense)"))?;
            vocab.extend_from_slice(t.as_bytes());
            off.push(vocab.len() as u32);
            token_ids.insert(t.to_string(), i as u32);
        }
        let mut small = vec![NONE; 0x200];
        for (i, t) in toks.iter().enumerate() {
            let mut it = t.expect("dense").chars();
            if let (Some(c), None) = (it.next(), it.next())
                && (c as u32) < 0x200
            {
                small[c as usize] = i as u32;
            }
        }

        let list = model.get("merges").and_then(Value::as_array).ok_or_else(|| bad("merges"))?;
        let mut merges = IdMap::with_capacity_and_hasher(list.len(), Default::default());
        for (rank, m) in list.iter().enumerate() {
            let (a, b) = match m {
                Value::Str(p) => p.split_once(' ').ok_or_else(|| bad("merge"))?,
                Value::Array(p) if p.len() == 2 => {
                    (p[0].as_str().ok_or_else(|| bad("merge"))?, p[1].as_str().ok_or_else(|| bad("merge"))?)
                }
                _ => return Err(bad("merge")),
            };
            let id = |t: &str| token_ids.get(t).copied().ok_or_else(|| bad("merge (token outside the vocabulary)"));
            let merged = id(&format!("{a}{b}"))?;
            merges.insert((id(a)? as u64) << 32 | id(b)? as u64, (rank as u32, merged));
        }

        let mut added = Vec::new();
        let mut added_by_id = HashMap::new();
        for a in spec.get("added_tokens").and_then(Value::as_array).unwrap_or(&[]) {
            let id = a.get("id").and_then(Value::as_i64).ok_or_else(|| bad("added token"))? as u32;
            let content = s(a.get("content")).filter(|c| !c.is_empty()).ok_or_else(|| bad("added token"))?;
            let flag = |k: &str| a.get(k).and_then(Value::as_bool).unwrap_or(false);
            if flag("normalized") || flag("lstrip") || flag("rstrip") || flag("single_word") {
                return Err(bad("added token options"));
            }
            added_by_id.insert(id, (content.clone(), flag("special")));
            added.push((content.into_bytes(), id));
        }

        let sym = byte_symbols();
        let unsym = sym.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Ok(Tokenizer { vocab, off, small, merges, added: AddedSet::build(added), added_by_id, token_ids, sym, unsym })
    }

    /// A token's id: a model-vocabulary token (GPT-2 symbols) or an added
    /// one (its content).
    pub fn token_id(&self, t: &str) -> Option<u32> {
        self.token_ids.get(t).copied().or_else(|| self.added_by_id.iter().find(|(_, (c, _))| c == t).map(|(&i, _)| i))
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3);
        let mut pieces = Vec::new();
        let mut word = Word::default();
        let mut mapped = String::new();
        self.added.split(text, &mut pieces);
        for (piece, id) in pieces {
            if let Some(id) = id {
                ids.push(id);
                continue;
            }
            let normalized = unicode::nfc_hf(piece);
            qwen_split(&normalized, |w| {
                mapped.clear();
                mapped.extend(w.bytes().map(|b| self.sym[b as usize]));
                self.bpe(&mapped, &mut word, &mut ids);
            });
        }
        ids
    }

    /// The bytes of tokens `ids` (special added tokens skipped when
    /// `skip_special`), as `tok.decode` joins them before its UTF-8 decode.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some((content, special)) = self.added_by_id.get(&id) {
                if !(skip_special && *special) {
                    out.extend_from_slice(content.as_bytes());
                }
                continue;
            }
            let i = id as usize;
            if i + 1 >= self.off.len() {
                continue;
            }
            let t = std::str::from_utf8(&self.vocab[self.off[i] as usize..self.off[i + 1] as usize]).expect("UTF-8");
            out.extend(t.chars().filter_map(|c| self.unsym.get(&c)));
        }
        out
    }

    /// `tok.decode(ids, skip_special_tokens=skip_special)` (invalid UTF-8
    /// replaced by U+FFFD, as Python's `errors="replace"`).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }

    fn bpe(&self, w: &str, word: &mut Word, out: &mut Vec<u32>) {
        word.clear();
        for c in w.chars() {
            // every GPT-2 symbol is in the vocabulary
            let id = self.small.get(c as usize).copied().unwrap_or(NONE);
            if id != NONE {
                word.add(id, c.len_utf8());
            }
        }
        word.merge_all(&self.merges);
        out.extend(word.syms.iter().filter(|s| s.len != 0).map(|s| s.id));
    }
}

#[derive(PartialEq, Clone, Copy)]
enum K {
    /// `\p{L}`
    L,
    /// `\p{M}`
    M,
    /// `\p{N}`
    N,
    /// `\s` other than CR / LF
    S,
    /// CR or LF
    Nl,
    /// anything else
    P,
}

fn kind(c: char) -> K {
    if c.is_ascii() {
        match c {
            'a'..='z' | 'A'..='Z' => K::L,
            '0'..='9' => K::N,
            '\r' | '\n' => K::Nl,
            '\t' | '\u{b}' | '\u{c}' | ' ' => K::S,
            _ => K::P,
        }
    } else if unicode::onig_letter(c) {
        K::L
    } else if unicode::onig_number(c) {
        K::N
    } else if unicode::is_white_space(c) {
        K::S
    } else if unicode::category(c).starts_with('M') {
        K::M
    } else {
        K::P
    }
}

/// Qwen's pre-tokenizer regex ([`QWEN_PATTERN`]), scanned by hand with
/// Oniguruma's semantics: ordered alternation, greedy runs with
/// backtracking, `\s` = White_Space. `f` gets each piece in order.
pub fn qwen_split(s: &str, mut f: impl FnMut(&str)) {
    let cs: Vec<(usize, char, K)> = s.char_indices().map(|(i, c)| (i, c, kind(c))).collect();
    let n = cs.len();
    let at = |j: usize| cs.get(j).map_or(s.len(), |x| x.0);
    let run = |mut j: usize, ok: &dyn Fn(K) -> bool| {
        while j < n && ok(cs[j].2) {
            j += 1;
        }
        j
    };
    let lm = |k: K| matches!(k, K::L | K::M);
    let space = |k: K| matches!(k, K::S | K::Nl);
    let mut i = 0;
    while i < n {
        let (c0, k0) = (cs[i].1, cs[i].2);
        let k1 = cs.get(i + 1).map(|x| x.2);
        let end = if let Some(len) = contraction(&s[cs[i].0..]) {
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            i + len
        } else if !matches!(k0, K::Nl | K::L | K::N) && k1.is_some_and(lm) {
            // [^\r\n\p{L}\p{N}]? taking one character, then [\p{L}\p{M}]+
            run(i + 1, &lm)
        } else if lm(k0) {
            run(i, &lm)
        } else if k0 == K::N {
            // \p{N}
            i + 1
        } else if (c0 == ' ' && k1 == Some(K::P)) || k0 == K::P {
            //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let j = run(if k0 == K::P { i } else { i + 1 }, &|k| k == K::P);
            run(j, &|k| k == K::Nl)
        } else {
            // whitespace: \s*[\r\n]+ ends after the run's last newline;
            // else \s+(?!\S) leaves the last character for the text that
            // follows (a lone one is \s+)
            let e = run(i, &space);
            match (i..e).rev().find(|&j| cs[j].2 == K::Nl) {
                Some(j) => j + 1,
                None if e < n && e - i >= 2 => e - 1,
                None => e,
            }
        };
        f(&s[cs[i].0..at(end)]);
        i = end;
    }
}

/// The length in bytes of a `'s|'t|'re|'ve|'m|'ll|'d` match (ASCII case
/// folded) at the start of `s`.
fn contraction(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'\'') {
        return None;
    }
    let low = |i: usize| b.get(i).map(u8::to_ascii_lowercase);
    for alt in ["s", "t", "re", "ve", "m", "ll", "d"] {
        if alt.bytes().enumerate().all(|(j, c)| low(1 + j) == Some(c)) {
            return Some(1 + alt.len());
        }
    }
    None
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    len: usize,
}

/// HF's `Word`, reused across pieces.
#[derive(Default)]
struct Word {
    syms: Vec<Symbol>,
    queue: BinaryHeap<Reverse<(u32, usize, u32)>>,
}

impl Word {
    fn clear(&mut self) {
        self.syms.clear();
        self.queue.clear();
    }

    fn add(&mut self, id: u32, len: usize) {
        let n = self.syms.len() as isize;
        if let Some(last) = self.syms.last_mut() {
            last.next = n;
        }
        self.syms.push(Symbol { id, prev: n - 1, next: -1, len });
    }

    /// `Word::merge_all` without dropout: a min-queue of (rank, position,
    /// merged id); an entry is stale when its position was absorbed or its
    /// pair no longer yields that id.
    fn merge_all(&mut self, merges: &IdMap<u64, (u32, u32)>) {
        let pair = |a: u32, b: u32| merges.get(&((a as u64) << 32 | b as u64)).copied();
        for i in 1..self.syms.len() {
            if let Some((rank, new)) = pair(self.syms[i - 1].id, self.syms[i].id) {
                self.queue.push(Reverse((rank, i - 1, new)));
            }
        }
        while let Some(Reverse((_, pos, new))) = self.queue.pop() {
            let cur = self.syms[pos];
            if cur.len == 0 || cur.next == -1 {
                continue;
            }
            let next = cur.next as usize;
            let right = self.syms[next];
            if pair(cur.id, right.id).is_none_or(|(_, id)| id != new) {
                continue;
            }
            self.syms[pos] = Symbol { id: new, prev: cur.prev, next: right.next, len: cur.len + right.len };
            self.syms[next].len = 0;
            if right.next >= 0 {
                self.syms[right.next as usize].prev = pos as isize;
            }
            if cur.prev >= 0 {
                let p = cur.prev as usize;
                if let Some((rank, id)) = pair(self.syms[p].id, new) {
                    self.queue.push(Reverse((rank, p, id)));
                }
            }
            let after = if right.next >= 0 { pair(new, self.syms[right.next as usize].id) } else { None };
            if let Some((rank, id)) = after {
                self.queue.push(Reverse((rank, pos, id)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::qwen_split;

    fn pieces(s: &str) -> Vec<String> {
        let mut v = Vec::new();
        qwen_split(s, |p| v.push(p.to_string()));
        v
    }

    #[test]
    fn splits_like_the_regex() {
        assert_eq!(pieces("Hello world!"), ["Hello", " world", "!"]);
        assert_eq!(pieces("I'm she'LL"), ["I", "'m", " she", "'LL"]);
        assert_eq!(pieces("x 123"), ["x", " ", "1", "2", "3"]);
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("a\n\n  b"), ["a", "\n\n", " ", " b"]);
        assert_eq!(pieces("end.\n\n"), ["end", ".\n\n"]);
        assert_eq!(pieces("  "), ["  "]);
        assert_eq!(pieces(" !!"), [" !!"]);
        assert_eq!(pieces("(hi)"), ["(hi", ")"]);
    }
}
