// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Reads the bundles IRON's exporters write for the Rust runtimes.
//!
//! IRON compiles ahead of time: an exporter (`iron/common/bundle.py` has the
//! writer side) builds every NPU kernel a model needs and writes a bundle
//! directory a Rust runtime replays:
//!
//! ```text
//! manifest.txt   one record a line: `<tag> <field>...`; a field `k=v` is
//!                also looked up by key. Blank lines and `#` lines are skipped.
//! tensors.txt    <name> <dtype> <d0,d1,...> <offset> <bytes>, one a line,
//! tensors.bin    into this blob (offsets 64-byte aligned; dtypes f32, bf16,
//!                u8, i32, little-endian)
//! kernels/       xclbins and instruction streams
//! ```
//!
//! Three manifest records mean the same thing in every bundle, and
//! [`Manifest`] interprets them:
//!
//! ```text
//! version <n>                        format version; must match the runtime's
//! param <name> <value...>            a model constant
//! xclbin <key> <file> <kernel name>  a hardware context kernels refer to by key
//! ```
//!
//! Every other record (kernels, steps, test cases) belongs to the model, and
//! its runtime reads them from [`Manifest::records`], in file order.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug)]
pub enum Error {
    /// A bundle file could not be read.
    Io(PathBuf, std::io::Error),
    /// A bundle file is malformed or inconsistent (the message says where).
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Error::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

fn bad(what: impl Into<String>) -> Error {
    Error::Format(what.into())
}

// ----------------------------------------------------------------------------
// manifest.txt
// ----------------------------------------------------------------------------

/// One manifest line: a tag, then whitespace-separated fields.
#[derive(Debug, Clone)]
pub struct Record {
    /// 1-based line number in manifest.txt.
    pub line: usize,
    pub tag: String,
    /// Every field after the tag, `k=v` ones included.
    pub fields: Vec<String>,
    kv: HashMap<String, String>,
    text: String,
}

impl Record {
    fn parse(line: usize, text: &str) -> Option<Record> {
        let mut it = text.split_whitespace();
        let tag = it.next().filter(|t| !t.starts_with('#'))?.to_string();
        let fields: Vec<String> = it.map(str::to_string).collect();
        let kv = fields.iter().filter_map(|f| f.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Some(Record { line, tag, fields, kv, text: text.to_string() })
    }

    /// The line as written.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// An error pointing at this line.
    pub fn error(&self, msg: impl fmt::Display) -> Error {
        bad(format!("{}:{}: {msg}, in: {}", Manifest::FILE, self.line, self.text))
    }

    /// The `i`th field after the tag.
    pub fn field(&self, i: usize) -> Result<&str, Error> {
        self.fields.get(i).map(String::as_str).ok_or_else(|| self.error(format!("expected at least {} fields", i + 1)))
    }

    pub fn field_as<T: FromStr>(&self, i: usize) -> Result<T, Error> {
        let f = self.field(i)?;
        f.parse().map_err(|_| self.error(format!("field {} ({f}) is malformed", i + 1)))
    }

    /// The text after the first `n` fields (after the tag), spacing inside
    /// it intact -- for a free-text last field, such as a prompt.
    pub fn rest(&self, n: usize) -> String {
        let mut s = self.text.as_str();
        for _ in 0..=n {
            s = s.trim_start();
            s = s.find(char::is_whitespace).map_or("", |i| &s[i..]);
        }
        s.strip_prefix(' ').unwrap_or(s).to_string()
    }

    pub fn has(&self, key: &str) -> bool {
        self.kv.contains_key(key)
    }

    /// The value of field `key=`.
    pub fn str(&self, key: &str) -> Result<&str, Error> {
        self.kv.get(key).map(String::as_str).ok_or_else(|| self.error(format!("`{key}=` missing")))
    }

    pub fn get<T: FromStr>(&self, key: &str) -> Result<T, Error> {
        let v = self.str(key)?;
        v.parse().map_err(|_| self.error(format!("`{key}={v}` is malformed")))
    }

    /// A `key=0` / `key=1` field.
    pub fn flag(&self, key: &str) -> Result<bool, Error> {
        match self.str(key)? {
            "0" => Ok(false),
            "1" => Ok(true),
            v => Err(self.error(format!("`{key}={v}` is not 0 or 1"))),
        }
    }
}

/// A hardware context: an xclbin and the kernel name inside it.
#[derive(Debug, Clone)]
pub struct Xclbin {
    pub key: String,
    pub path: PathBuf,
    pub kernel: String,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub dir: PathBuf,
    pub version: u32,
    records: Vec<Record>,
    params: HashMap<String, String>,
    xclbins: HashMap<String, Xclbin>,
}

impl Manifest {
    pub const FILE: &'static str = "manifest.txt";

    /// Read `<dir>/manifest.txt`, which must be format `version`.
    pub fn load(dir: &Path, version: u32) -> Result<Self, Error> {
        let path = dir.join(Self::FILE);
        let text = fs::read_to_string(&path).map_err(|e| Error::Io(path.clone(), e))?;
        let mut m = Manifest {
            dir: dir.to_path_buf(),
            version: 0,
            records: Vec::new(),
            params: HashMap::new(),
            xclbins: HashMap::new(),
        };
        let mut seen_version = None;
        for (i, line) in text.lines().enumerate() {
            let Some(r) = Record::parse(i + 1, line) else { continue };
            match r.tag.as_str() {
                "version" => {
                    if seen_version.is_some() {
                        return Err(r.error("second version record"));
                    }
                    seen_version = Some(r.field_as::<u32>(0)?);
                }
                "param" => {
                    let k = r.field(0)?.to_string();
                    r.field(1)?;
                    if m.params.insert(k.clone(), r.rest(1)).is_some() {
                        return Err(r.error(format!("param {k} defined twice")));
                    }
                }
                "xclbin" => {
                    let x = Xclbin {
                        key: r.field(0)?.to_string(),
                        path: dir.join(r.field(1)?),
                        kernel: r.field(2)?.to_string(),
                    };
                    if m.xclbins.contains_key(&x.key) {
                        return Err(r.error(format!("xclbin {} defined twice", x.key)));
                    }
                    m.xclbins.insert(x.key.clone(), x);
                }
                _ => m.records.push(r),
            }
        }
        match seen_version {
            Some(v) if v == version => m.version = v,
            v => {
                let found = v.map_or("no version record".to_string(), |v| format!("version {v}"));
                return Err(bad(format!(
                    "{}: {found}, this runtime reads version {version}; re-export the bundle",
                    path.display()
                )));
            }
        }
        Ok(m)
    }

    /// Every model-specific record (not `version` / `param` / `xclbin`), in
    /// file order.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn tagged<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Record> {
        self.records.iter().filter(move |r| r.tag == tag)
    }

    /// A bundle-relative path.
    pub fn path(&self, rel: &str) -> PathBuf {
        self.dir.join(rel)
    }

    pub fn has_param(&self, k: &str) -> bool {
        self.params.contains_key(k)
    }

    pub fn param(&self, k: &str) -> Result<&str, Error> {
        self.params.get(k).map(String::as_str).ok_or_else(|| bad(format!("param {k} missing from the manifest")))
    }

    pub fn param_as<T: FromStr>(&self, k: &str) -> Result<T, Error> {
        let v = self.param(k)?;
        v.parse().map_err(|_| bad(format!("param {k} ({v}) is malformed")))
    }

    /// A comma-separated param (empty items skipped, so `""` is `[]`).
    pub fn list<T: FromStr>(&self, k: &str) -> Result<Vec<T>, Error> {
        self.param(k)?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().map_err(|_| bad(format!("param {k}: bad list item {s}"))))
            .collect()
    }

    pub fn xclbin(&self, key: &str) -> Result<&Xclbin, Error> {
        self.xclbins.get(key).ok_or_else(|| bad(format!("xclbin {key} missing from the manifest")))
    }

    pub fn xclbins(&self) -> impl Iterator<Item = &Xclbin> {
        self.xclbins.values()
    }
}

// ----------------------------------------------------------------------------
// tensors.txt + tensors.bin
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    U8,
    I32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::Bf16 => 2,
            DType::U8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    off: usize,
    len: usize,
}

impl Entry {
    /// Number of elements.
    pub fn elems(&self) -> usize {
        self.len / self.dtype.size()
    }
}

/// Every tensor of the bundle. On Unix `tensors.bin` is memory-mapped
/// read-only: a tensor's pages are read when it is first touched and stay
/// reclaimable page cache, so loading a multi-GB bundle costs no heap and
/// a runtime that uploads its weights and drops the store never holds them
/// twice. Elsewhere (or if mapping fails) the file is read into a `u64`
/// buffer. Either way the 64-byte-aligned offsets `tensors.txt` records are
/// aligned in memory (mappings are page-aligned), so the typed views below
/// are sound. Don't rewrite a bundle's `tensors.bin` while a runtime has it
/// loaded: exporters write new directories.
pub struct Store {
    data: Backing,
    map: HashMap<String, Entry>,
}

enum Backing {
    Heap(Vec<u64>),
    #[cfg(unix)]
    Mapped(mmap::Map),
}

impl Backing {
    fn as_ptr(&self) -> *const u8 {
        match self {
            Backing::Heap(v) => v.as_ptr() as *const u8,
            #[cfg(unix)]
            Backing::Mapped(m) => m.ptr,
        }
    }
}

#[cfg(unix)]
mod mmap {
    //! A read-only private file mapping, through libc's `mmap` (std has no
    //! wrapper; the constants are the same on Linux and macOS).
    use std::ffi::{c_int, c_void};
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;

    pub struct Map {
        pub ptr: *const u8,
        len: usize,
    }

    // SAFETY: the mapping is read-only and owned by the Map for its life.
    unsafe impl Send for Map {}
    unsafe impl Sync for Map {}

    impl Map {
        /// `file`'s first `len` bytes (`len > 0`), or None if mmap fails.
        pub fn new(file: &std::fs::File, len: usize) -> Option<Map> {
            // SAFETY: a fresh read-only mapping of a file we hold open; the
            // result is checked against MAP_FAILED (-1).
            let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
            if p as isize == -1 { None } else { Some(Map { ptr: p as *const u8, len }) }
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            // SAFETY: the mapping this Map created, unmapped once.
            unsafe { munmap(self.ptr as *mut c_void, self.len) };
        }
    }
}

impl Store {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let idx = dir.join("tensors.txt");
        let text = fs::read_to_string(&idx).map_err(|e| Error::Io(idx.clone(), e))?;
        let mut map = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let at = |msg: &str| bad(format!("tensors.txt:{}: {msg}, in: {line}", i + 1));
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 5 {
                return Err(at("expected <name> <dtype> <shape> <offset> <bytes>"));
            }
            let dtype = match f[1] {
                "f32" => DType::F32,
                "bf16" => DType::Bf16,
                "u8" => DType::U8,
                "i32" => DType::I32,
                _ => return Err(at("unknown dtype")),
            };
            let shape: Vec<usize> =
                f[2].split(',').map(|s| s.parse().map_err(|_| at("bad shape"))).collect::<Result<_, _>>()?;
            let off: usize = f[3].parse().map_err(|_| at("bad offset"))?;
            let len: usize = f[4].parse().map_err(|_| at("bad byte count"))?;
            if !off.is_multiple_of(8) {
                return Err(at("offset is not 8-byte aligned"));
            }
            if !len.is_multiple_of(dtype.size()) {
                return Err(at("byte count is not a whole number of elements"));
            }
            if map.insert(f[0].to_string(), Entry { dtype, shape, off, len }).is_some() {
                return Err(at("tensor defined twice"));
            }
        }
        let bin = dir.join("tensors.bin");
        let mut file = fs::File::open(&bin).map_err(|e| Error::Io(bin.clone(), e))?;
        let bytes = file.metadata().map_err(|e| Error::Io(bin.clone(), e))?.len() as usize;
        #[cfg(unix)]
        let mapped = if bytes > 0 { mmap::Map::new(&file, bytes).map(Backing::Mapped) } else { None };
        #[cfg(not(unix))]
        let mapped = None;
        let data = match mapped {
            Some(m) => m,
            None => {
                let mut data = vec![0u64; bytes.div_ceil(8)];
                // SAFETY: a u64 buffer viewed as its bytes.
                let raw = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, bytes) };
                file.read_exact(raw).map_err(|e| Error::Io(bin.clone(), e))?;
                Backing::Heap(data)
            }
        };
        for (name, e) in &map {
            if e.off + e.len > bytes {
                return Err(bad(format!("tensor {name} runs past the end of tensors.bin")));
            }
        }
        Ok(Store { data, map })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    pub fn entry(&self, name: &str) -> Result<&Entry, Error> {
        self.map.get(name).ok_or_else(|| bad(format!("tensor {name} missing from the bundle")))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(&self.entry(name)?.shape)
    }

    /// `name`'s entry, checked to be `dtype` with `elems` elements.
    pub fn expect(&self, name: &str, dtype: DType, elems: usize) -> Result<&Entry, Error> {
        let e = self.entry(name)?;
        if e.dtype != dtype || e.elems() != elems {
            return Err(bad(format!("tensor {name} is {} x {:?}, wanted {elems} x {dtype:?}", e.elems(), e.dtype)));
        }
        Ok(e)
    }

    fn typed<T: Copy>(&self, name: &str, want: DType) -> Result<&[T], Error> {
        let e = self.entry(name)?;
        if e.dtype != want {
            return Err(bad(format!("tensor {name} is {:?}, wanted {want:?}", e.dtype)));
        }
        // SAFETY: in bounds (checked at load), aligned (8-byte offsets into a
        // u64 buffer or a page-aligned mapping), and every bit pattern is a
        // valid f32/u16/u8/i32.
        Ok(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr().add(e.off) as *const T, e.len / std::mem::size_of::<T>())
        })
    }

    /// Any tensor's raw bytes.
    pub fn bytes(&self, name: &str) -> Result<&[u8], Error> {
        let e = self.entry(name)?;
        // SAFETY: in bounds (checked at load).
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(e.off), e.len) })
    }

    pub fn f32(&self, name: &str) -> Result<&[f32], Error> {
        self.typed(name, DType::F32)
    }

    /// bf16 as raw bits.
    pub fn bf16(&self, name: &str) -> Result<&[u16], Error> {
        self.typed(name, DType::Bf16)
    }

    pub fn u8(&self, name: &str) -> Result<&[u8], Error> {
        self.typed(name, DType::U8)
    }

    pub fn i32(&self, name: &str) -> Result<&[i32], Error> {
        self.typed(name, DType::I32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn tmpdir() -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "taconite-bundle-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn manifest(text: &str) -> Result<Manifest, Error> {
        let d = tmpdir();
        fs::write(d.join("manifest.txt"), text).unwrap();
        Manifest::load(&d, 3)
    }

    #[test]
    fn records_params_xclbins() {
        let m = manifest(
            "version 3\n\
             # a comment\n\
             \n\
             param grid 72\n\
             param splits 1,2,,3\n\
             param name IR 18\n\
             xclbin ctx0 kernels/a.xclbin MLIR_AIE\n\
             gemm g0 ctx=ctx0 M=256 bias=1\n\
             case 0 cases/cats.jpg Two  dogs, playing!\n",
        )
        .unwrap();
        assert_eq!(m.param_as::<usize>("grid").unwrap(), 72);
        assert_eq!(m.list::<usize>("splits").unwrap(), [1, 2, 3]);
        assert_eq!(m.param("name").unwrap(), "IR 18");
        let x = m.xclbin("ctx0").unwrap();
        assert_eq!((x.path.ends_with("kernels/a.xclbin"), x.kernel.as_str()), (true, "MLIR_AIE"));
        assert_eq!(m.records().len(), 2);
        let g = m.tagged("gemm").next().unwrap();
        assert_eq!((g.line, g.field(0).unwrap(), g.str("ctx").unwrap()), (8, "g0", "ctx0"));
        assert_eq!(g.get::<usize>("M").unwrap(), 256);
        assert!(g.flag("bias").unwrap());
        assert!(g.get::<usize>("N").unwrap_err().to_string().starts_with("manifest.txt:8: `N=` missing"));
        let c = m.tagged("case").next().unwrap();
        assert_eq!(c.rest(2), "Two  dogs, playing!");
        assert_eq!(c.rest(0), "0 cases/cats.jpg Two  dogs, playing!");
    }

    #[test]
    fn version_is_checked() {
        let e = manifest("version 2\n").unwrap_err().to_string();
        assert!(e.contains("version 2, this runtime reads version 3"), "{e}");
        let e = manifest("param a 1\n").unwrap_err().to_string();
        assert!(e.contains("no version record"), "{e}");
        assert!(manifest("version 3\nversion 3\n").is_err());
        assert!(manifest("version 3\nparam a 1\nparam a 2\n").is_err());
        assert!(manifest("version 3\nxclbin k f\n").is_err());
    }

    fn write_store(d: &Path, idx: &str, bin: &[u8]) -> Result<Store, Error> {
        fs::write(d.join("tensors.txt"), idx).unwrap();
        fs::write(d.join("tensors.bin"), bin).unwrap();
        Store::load(d)
    }

    #[test]
    fn store() {
        let d = tmpdir();
        let mut bin = vec![0u8; 72];
        bin[..8].copy_from_slice(&[0, 0, 128, 63, 0, 0, 0, 64]); // f32 1.0, 2.0
        bin[64..68].copy_from_slice(&[0x80, 0x3f, 0x00, 0x40]); // bf16 1.0, 2.0
        let s = write_store(&d, "a f32 2 0 8\nb bf16 1,2 64 4\n", &bin).unwrap();
        assert_eq!(s.f32("a").unwrap(), [1.0, 2.0]);
        assert_eq!(s.bf16("b").unwrap(), [0x3f80, 0x4000]);
        assert_eq!(s.shape("b").unwrap(), [1, 2]);
        assert!(s.f32("b").is_err());
        assert!(s.expect("b", DType::Bf16, 2).is_ok());
        assert!(s.expect("b", DType::Bf16, 3).is_err());
        assert!(s.entry("c").is_err());
        assert!(write_store(&d, "a f32 2 0 8\nb bf16 1,6 64 12\n", &bin).is_err()); // past the end
        assert!(write_store(&d, "a f32 2 4 8\n", &bin).is_err()); // misaligned
        assert!(write_store(&d, "a f32 2 0 6\n", &bin).is_err()); // partial element
        assert!(write_store(&d, "a f32 2 0 8\na f32 2 0 8\n", &bin).is_err()); // twice
    }
}
