// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

/// Splits `out` (rows of `row` elements) over up to `threads` scoped
/// threads; `f(first_row, rows)` fills each piece.
pub(crate) fn par_rows<T: Send>(out: &mut [T], row: usize, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / row.max(1);
    if rows == 0 {
        return;
    }
    let per = rows.div_ceil(threads.max(1));
    if per >= rows {
        f(0, out);
        return;
    }
    std::thread::scope(|s| {
        for (i, piece) in out.chunks_mut(per * row).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, piece));
        }
    });
}

/// The host threads to use by default: the machine's, up to 16.
pub(crate) fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(16)
}
