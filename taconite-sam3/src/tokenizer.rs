// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLIP's byte-level BPE tokenizer (HF `CLIPTokenizerFast`), zero-dependency.
//!
//! Normalizer: whitespace runs -> one space, then lowercase (HF also applies
//! NFC first; std has no Unicode normalization, so composed input is
//! assumed -- what every keyboard and most text produce). Pre-tokenizer: the
//! regex `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`,
//! matches kept and whitespace dropped, scanned by hand with std's
//! Unicode-aware `char` predicates; each piece is mapped to GPT-2's
//! byte-to-unicode surface and BPE-merged with `</w>` on its last symbol.
//! Then `<|startoftext|> ... <|endoftext|>`, truncated and padded to the
//! model's 32 tokens.

use std::collections::HashMap;
use std::path::Path;

use crate::Error;

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

/// A flat `{"token": id}` JSON object (the `vocab.json` HF writes): string
/// keys with `\"`, `\\`, `\/`, `\n`-style and `\uXXXX` (incl. surrogate
/// pair) escapes, integer values.
fn parse_vocab(text: &str) -> Result<HashMap<String, u32>, Error> {
    let bad = || Error::Bundle("vocab.json: not a flat {string: int} object".into());
    let c: Vec<char> = text.chars().collect();
    let mut i = c.iter().position(|&x| x == '{').ok_or_else(bad)? + 1;
    let mut map = HashMap::new();
    let hex4 = |c: &[char], i: usize| -> Option<u32> {
        let s: String = c.get(i..i + 4)?.iter().collect();
        u32::from_str_radix(&s, 16).ok()
    };
    loop {
        while i < c.len() && (c[i].is_whitespace() || c[i] == ',') {
            i += 1;
        }
        if i >= c.len() || c[i] == '}' {
            break;
        }
        if c[i] != '"' {
            return Err(bad());
        }
        i += 1;
        let mut key = String::new();
        while i < c.len() && c[i] != '"' {
            if c[i] == '\\' {
                i += 1;
                match c.get(i) {
                    Some('n') => key.push('\n'),
                    Some('t') => key.push('\t'),
                    Some('r') => key.push('\r'),
                    Some('b') => key.push('\u{8}'),
                    Some('f') => key.push('\u{c}'),
                    Some('u') => {
                        let mut u = hex4(&c, i + 1).ok_or_else(bad)?;
                        i += 4;
                        if (0xD800..0xDC00).contains(&u) && c.get(i + 1) == Some(&'\\') && c.get(i + 2) == Some(&'u') {
                            let lo = hex4(&c, i + 3).ok_or_else(bad)?;
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                        key.push(char::from_u32(u).ok_or_else(bad)?);
                    }
                    Some(&o) => key.push(o),
                    None => return Err(bad()),
                }
            } else {
                key.push(c[i]);
            }
            i += 1;
        }
        i += 1;
        while i < c.len() && (c[i].is_whitespace() || c[i] == ':') {
            i += 1;
        }
        let start = i;
        while i < c.len() && c[i].is_ascii_digit() {
            i += 1;
        }
        let num: String = c[start..i].iter().collect();
        map.insert(key, num.parse().map_err(|_| bad())?);
    }
    Ok(map)
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    table: [char; 256],
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub max_len: usize,
}

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// The regex's matches, in order (whitespace between them dropped).
fn pretokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let other = |x: char| !x.is_whitespace() && !x.is_alphabetic() && !x.is_numeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let x = c[i];
        if x.is_whitespace() {
            i += 1;
            continue;
        }
        // alternatives in the regex's order, at this position
        if x == '\'' {
            if let Some(k) = CONTRACTIONS.iter().find(|k| {
                let kc: Vec<char> = k.chars().collect();
                c.len() >= i + kc.len() && c[i..i + kc.len()] == kc[..]
            }) {
                out.push(k.to_string());
                i += k.chars().count();
                continue;
            }
        }
        let start = i;
        if x.is_alphabetic() {
            while i < c.len() && c[i].is_alphabetic() {
                i += 1;
            }
        } else if x.is_numeric() {
            i += 1;
        } else {
            while i < c.len() && other(c[i]) {
                i += 1;
            }
        }
        out.push(c[start..i].iter().collect());
    }
    out
}

impl Tokenizer {
    pub fn load(dir: &Path, bos: u32, eos: u32, pad: u32, max_len: usize) -> Result<Self, Error> {
        let read = |f: &str| {
            std::fs::read_to_string(dir.join(f)).map_err(|e| Error::Bundle(format!("{}: {e}", dir.join(f).display())))
        };
        let vocab = parse_vocab(&read("vocab.json")?)?;
        let mut ranks = HashMap::new();
        for line in read("merges.txt")?.lines() {
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let (a, b) = line.split_once(' ').ok_or_else(|| Error::Bundle(format!("merges.txt: {line}")))?;
            let r = ranks.len();
            ranks.insert((a.to_string(), b.to_string()), r);
        }
        Ok(Tokenizer { vocab, ranks, table: bytes_to_unicode(), bos, eos, pad, max_len })
    }

    fn bpe(&self, piece: &str) -> Vec<String> {
        let surface: String = piece.bytes().map(|b| self.table[b as usize]).collect();
        let mut word: Vec<String> = surface.chars().map(String::from).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while word.len() > 1 {
            let best = (0..word.len() - 1)
                .filter_map(|i| self.ranks.get(&(word[i].clone(), word[i + 1].clone())).map(|&r| (r, i)))
                .min();
            let Some((_, idx)) = best else { break };
            let (a, b) = (word[idx].clone(), word[idx + 1].clone());
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    /// `(input_ids, attention_mask)`, both `max_len` long.
    pub fn encode(&self, text: &str) -> (Vec<u32>, Vec<u32>) {
        let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![self.bos];
        'outer: for piece in pretokenize(&normalized) {
            for tok in self.bpe(&piece) {
                if ids.len() == self.max_len - 1 {
                    break 'outer;
                }
                ids.push(*self.vocab.get(&tok).unwrap_or(&self.eos));
            }
        }
        ids.push(self.eos);
        let n = ids.len();
        let mut mask = vec![1; n];
        ids.resize(self.max_len, self.pad);
        mask.resize(self.max_len, 0);
        (ids, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenizer_follows_the_clip_regex() {
        assert_eq!(pretokenize("the laptop's screen"), ["the", "laptop", "'s", "screen"]);
        assert_eq!(pretokenize("two dogs, playing!"), ["two", "dogs", ",", "playing", "!"]);
        assert_eq!(pretokenize("red car 22"), ["red", "car", "2", "2"]);
        assert_eq!(pretokenize("a!'s"), ["a", "!'", "s"]);
    }

    #[test]
    fn vocab_parser_handles_escapes() {
        let v = parse_vocab(r#"{"a": 1, "\"q\"": 2, "été</w>": 3, "😀": 4}"#).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["\"q\""], 2);
        assert_eq!(v["été</w>"], 3);
        assert_eq!(v["😀"], 4);
    }
}
