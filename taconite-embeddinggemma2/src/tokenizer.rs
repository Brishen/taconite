// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! EmbeddingGemma 2's tokenizer (Gemma's `tokenizer.json`), as Hugging
//! Face `tokenizers` runs it and sentence-transformers calls it:
//!
//! 1. the added tokens (`<bos>`, `<eos>`, `<mask>`, `<|image|>`, ...) are
//!    split out of the text first, leftmost-longest, and map to their ids;
//! 2. every other piece is normalized (each space becomes U+2581 `▁`) and
//!    is one BPE word (the pre-tokenizer splits on spaces, which no longer
//!    exist): its characters are the initial symbols, a character outside
//!    the vocabulary becomes its UTF-8 bytes' `<0xXX>` tokens (byte
//!    fallback), and adjacent pairs merge lowest rank first, the leftmost
//!    of equal ranks first;
//! 3. `<bos> ... <eos>` around the ids -- unless the text already starts
//!    with `<bos>`, when transformers' chat-template path adds neither.
//!
//! The vocabulary and the merges come from the bundle (`tok.*` tensors,
//! written by `export_eg2.py`), so nothing parses the 32 MB JSON here.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use taconite_bundle::{Manifest, Store};

use crate::Error;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// (left, right) -> (rank, merged)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// added tokens, longest first
    added: Vec<(String, u32)>,
    bytes: [u32; 256],
    pub bos: u32,
    pub eos: u32,
    /// sentence-transformers' prompts by name (e.g. "SearchQuery" ->
    /// "task: search result | query: ")
    pub prompts: Vec<(String, String)>,
}

fn unhex(h: &str) -> Result<String, Error> {
    if h == "-" {
        return Ok(String::new());
    }
    let b: Result<Vec<u8>, _> = (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16)).collect();
    String::from_utf8(b.map_err(|_| Error::Bundle(format!("bad hex {h}")))?)
        .map_err(|_| Error::Bundle(format!("non-UTF-8 hex {h}")))
}

impl Tokenizer {
    /// None when the bundle has no tokenizer (an image-only export).
    pub fn load(m: &Manifest, s: &Store) -> Result<Option<Self>, Error> {
        if !s.has("tok.vocab") {
            return Ok(None);
        }
        let blob = s.u8("tok.vocab")?;
        let off = s.i32("tok.vocab_off")?;
        let mut vocab = HashMap::with_capacity(off.len());
        for i in 0..off.len() - 1 {
            let piece = std::str::from_utf8(&blob[off[i] as usize..off[i + 1] as usize])
                .map_err(|_| Error::Bundle(format!("token {i} is not UTF-8")))?;
            vocab.insert(piece.to_string(), i as u32);
        }
        let mg = s.i32("tok.merges")?;
        let mut merges = HashMap::with_capacity(mg.len() / 3);
        for (rank, t) in mg.chunks(3).enumerate() {
            merges.entry((t[0] as u32, t[1] as u32)).or_insert((rank as u32, t[2] as u32));
        }
        let mut added = Vec::new();
        for r in m.tagged("added") {
            added.push((unhex(r.field(1)?)?, r.field_as::<u32>(0)?));
        }
        added.sort_by_key(|(t, _)| Reverse(t.len()));
        let mut bytes = [0u32; 256];
        for (b, slot) in bytes.iter_mut().enumerate() {
            let k = format!("<0x{b:02X}>");
            *slot = *vocab.get(&k).ok_or_else(|| Error::Bundle(format!("no byte token {k}")))?;
        }
        let mut prompts = Vec::new();
        for r in m.tagged("prompt") {
            prompts.push((r.field(0)?.to_string(), unhex(r.field(1)?)?));
        }
        Ok(Some(Tokenizer {
            vocab,
            merges,
            added,
            bytes,
            bos: m.param_as("bos")?,
            eos: m.param_as("eos")?,
            prompts,
        }))
    }

    /// The prompt text for `name` (sentence-transformers' prompt names).
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.iter().find(|(n, _)| n == name).map(|(_, p)| p.as_str())
    }

    /// `text`'s ids as sentence-transformers makes them.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let special = !text.starts_with("<bos>");
        let mut ids = Vec::new();
        if special {
            ids.push(self.bos);
        }
        let mut rest = text;
        while !rest.is_empty() {
            // the next added token at or after the start of `rest`
            let mut hit: Option<(usize, usize, u32)> = None;
            for (pos, _) in rest.char_indices() {
                if let Some((t, id)) = self.added.iter().find(|(t, _)| rest[pos..].starts_with(t.as_str())) {
                    hit = Some((pos, t.len(), *id));
                    break;
                }
            }
            let (end, next) = match hit {
                Some((pos, len, id)) => (pos, Some((pos + len, id))),
                None => (rest.len(), None),
            };
            if end > 0 {
                self.word(&rest[..end].replace(' ', "\u{2581}"), &mut ids);
            }
            match next {
                Some((after, id)) => {
                    ids.push(id);
                    rest = &rest[after..];
                }
                None => break,
            }
        }
        if special {
            ids.push(self.eos);
        }
        ids
    }

    /// One BPE word's ids into `out`.
    fn word(&self, w: &str, out: &mut Vec<u32>) {
        // symbols as a linked list over a vector
        let mut ids: Vec<u32> = Vec::with_capacity(w.len());
        let mut buf = [0u8; 4];
        for ch in w.chars() {
            match self.vocab.get(ch.encode_utf8(&mut buf) as &str) {
                Some(&id) => ids.push(id),
                None => ids.extend(ch.to_string().bytes().map(|b| self.bytes[b as usize])),
            }
        }
        let n = ids.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut alive = vec![true; n];
        // (rank, position, left id, right id): stale entries are skipped
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<(u32, usize, u32, u32)>>, i: usize, a: u32, b: u32| {
            if let Some(&(rank, _)) = self.merges.get(&(a, b)) {
                heap.push(Reverse((rank, i, a, b)));
            }
        };
        for i in 0..n.saturating_sub(1) {
            push(&mut heap, i, ids[i], ids[i + 1]);
        }
        while let Some(Reverse((_, i, a, b))) = heap.pop() {
            let j = next[i];
            if !alive[i] || j >= n || !alive[j] || ids[i] != a || ids[j] != b {
                continue;
            }
            let merged = self.merges[&(a, b)].1;
            ids[i] = merged;
            alive[j] = false;
            next[i] = next[j];
            if next[i] < n {
                prev[next[i]] = i as isize;
                push(&mut heap, i, merged, ids[next[i]]);
            }
            if prev[i] >= 0 {
                let p = prev[i] as usize;
                push(&mut heap, p, ids[p], merged);
            }
        }
        let mut i = 0;
        while i < n {
            if alive[i] {
                out.push(ids[i]);
            }
            i = if alive[i] { next[i] } else { i + 1 };
        }
    }
}
