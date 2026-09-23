// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-FileCopyrightText: Copyright (c) 2026 Eugene Hauptmann, Nataliya Kosmyna (RLX)
// SPDX-License-Identifier: Apache-2.0

//! Building IRON kernels from Rust, with no Python: the two compile steps
//! IRON's Python drives (`iron/common/compilation`) as plain subprocesses.
//!
//! - [`Toolchain::compile_kernel`]: an AIE core kernel (`aie_kernels/…/*.cc`)
//!   to an object with Peano's `clang++`, the flags mlir-aie's
//!   `compile_cxx_core_function` and IRON's `KernelCompilationRule` pass.
//! - [`Toolchain::compile_design`]: a design's MLIR (what an operator's
//!   `design.py` generates) to the `.xclbin` + instruction stream
//!   [`Session::load_kernel`](crate::Session::load_kernel) replays, with the
//!   native `aiecc` binary — the one `aiecc.py` and mlir-aie's
//!   `compile_mlir_module` exec — and the flags the latter passes it.
//!
//! Generating the MLIR stays in Python (it is what `design.py` is); this is
//! for everything after, e.g. rebuilding a bundle's kernels on the machine
//! that runs them. The flags follow the mlir-aie these operators are built
//! with (`requirements.txt`): aiecc's output flags have been renamed across
//! releases (`--aie-generate-xclbin` → `--get-xclbin`), so a much older or
//! newer aiecc may reject them.
//!
//! Adapted from RLX's `rlx-xdna/src/compile.rs`
//! (<https://github.com/MIT-RLX/rlx>, MIT OR Apache-2.0), which found the
//! native aiecc is all a Python-free build needs; the stale-output and
//! link-object handling below come from there.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use crate::Error;

/// The AIE core architecture a kernel object is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// NPU1 (Phoenix, Hawk Point).
    Aie2,
    /// NPU2 (Strix, Strix Halo, Krackan).
    Aie2p,
}

impl Arch {
    /// Peano's name for it, as in `--target=aie2p-none-unknown-elf`.
    pub fn as_str(self) -> &'static str {
        match self {
            Arch::Aie2 => "aie2",
            Arch::Aie2p => "aie2p",
        }
    }

    /// The architecture of the device an MLIR design targets, read from its
    /// `aie.device(npu1…)` / `aie.device(npu2…)` — what the design's kernel
    /// objects have to be compiled for.
    pub fn of_mlir(mlir: &str) -> Option<Arch> {
        let rest = &mlir[mlir.find("aie.device(")? + "aie.device(".len()..];
        if rest.starts_with("npu1") {
            Some(Arch::Aie2)
        } else if rest.starts_with("npu2") {
            Some(Arch::Aie2p)
        } else {
            None
        }
    }

    /// The directory of `aie_runtime_lib` holding this arch's headers.
    fn runtime_lib_dir(self) -> &'static str {
        match self {
            Arch::Aie2 => "AIE2",
            Arch::Aie2p => "AIE2P",
        }
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the compilers are: an mlir-aie install (`bin/aiecc`, `include/`,
/// `aie_runtime_lib/`) and a Peano (`llvm-aie`) install.
#[derive(Debug, Clone)]
pub struct Toolchain {
    mlir_aie: PathBuf,
    aiecc: PathBuf,
    peano: PathBuf,
    launcher: Vec<OsString>,
}

impl Toolchain {
    /// The toolchain of the mlir-aie install at `mlir_aie` (the `mlir_aie`
    /// directory of the pip wheel, or `$MLIR_AIE_INSTALL_DIR`). Peano is
    /// looked for where mlir-aie itself looks: `$PEANO_INSTALL_DIR`, then
    /// `<mlir_aie>/peano`, then the `llvm-aie` wheel beside it.
    pub fn new(mlir_aie: impl Into<PathBuf>) -> Self {
        let mlir_aie = mlir_aie.into();
        let aiecc = mlir_aie.join("bin").join("aiecc");
        let peano = std::env::var_os("PEANO_INSTALL_DIR")
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| Some(mlir_aie.join("peano")).filter(|p| p.is_dir()))
            .unwrap_or_else(|| mlir_aie.parent().unwrap_or(Path::new("")).join("llvm-aie"));
        Self { mlir_aie, aiecc, peano, launcher: Vec::new() }
    }

    /// [`new`](Self::new) on `$MLIR_AIE_INSTALL_DIR` (what mlir-aie's
    /// `env_setup.sh` exports), with `$AIECC_PATH`, when set, naming the
    /// aiecc binary as it does for mlir-aie's own JIT.
    pub fn from_env() -> Result<Self, Error> {
        let root = std::env::var_os("MLIR_AIE_INSTALL_DIR").ok_or_else(|| {
            Error::Compile(
                "MLIR_AIE_INSTALL_DIR is not set (point it at the mlir_aie directory of the mlir-aie wheel)".into(),
            )
        })?;
        let mut tc = Self::new(root);
        if let Some(aiecc) = std::env::var_os("AIECC_PATH") {
            tc.aiecc = aiecc.into();
        }
        Ok(tc)
    }

    /// Uses `aiecc` as the aiecc binary: the native one (~200 MB ELF), not
    /// the `aiecc.py` wrapper script.
    pub fn with_aiecc(mut self, aiecc: impl Into<PathBuf>) -> Self {
        self.aiecc = aiecc.into();
        self
    }

    /// Uses the Peano install at `peano` (the directory holding `bin/clang++`).
    pub fn with_peano(mut self, peano: impl Into<PathBuf>) -> Self {
        self.peano = peano.into();
        self
    }

    /// Runs every tool as `launcher… tool args…`, e.g. through a script that
    /// execs its arguments inside a container. mlir-aie's binaries are built
    /// for a generic (FHS) Linux, so on a host that can't run them — NixOS —
    /// they run in a container with the same paths mounted; the paths given
    /// here are made absolute, so the container sees the same files.
    pub fn with_launcher<I, S>(mut self, launcher: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.launcher = launcher.into_iter().map(Into::into).collect();
        self
    }

    pub fn mlir_aie(&self) -> &Path {
        &self.mlir_aie
    }

    pub fn aiecc(&self) -> &Path {
        &self.aiecc
    }

    pub fn peano(&self) -> &Path {
        &self.peano
    }

    /// Compiles one AIE core kernel (C++, AIE API) to the object `output`
    /// (plus the dependency file `output.d`), the way IRON's
    /// `KernelCompilationRule` does with Peano.
    pub fn compile_kernel(&self, kernel: &KernelSource, output: &Path) -> Result<(), Error> {
        let source = absolute(&kernel.source)?;
        if !source.is_file() {
            return Err(Error::Compile(format!("kernel source {} does not exist", source.display())));
        }
        let output = absolute(output)?;
        if let Some(dir) = output.parent() {
            create_dir(dir)?;
        }
        remove_stale(&output)?;
        let dep = with_suffix(&output, ".d");
        let arch = kernel.arch;

        let mut cmd = self.command(&self.peano.join("bin").join("clang++"));
        cmd.arg(&source).arg("-c").arg("-o").arg(&output);
        cmd.arg(format!("-I{}", self.mlir_aie.join("include").display()));
        cmd.args([
            "-std=c++20",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            // aie_api tests capability macros Peano never defines, so -Wundef
            // only works with its headers treated as system headers.
            "--system-header-prefix=aie_api/",
            "-Werror=undef",
            "-O2",
            "-DNDEBUG",
            "-MD",
            "-MF",
        ]);
        cmd.arg(&dep);
        // Pre-trips aie_api's include guard for aie_adf.hpp, which would pull
        // in the Vitis-only <adf.h>.
        cmd.arg("-D__AIE_API_AIE_ADF_HPP__");
        cmd.arg(format!("--target={arch}-none-unknown-elf"));
        // The stack accounting aiecc's llc gives a core's own code.
        cmd.args(["-ffunction-sections", "-fdata-sections", "-fstack-size-section"]);
        cmd.arg("-I").arg(self.mlir_aie.join("aie_runtime_lib").join(arch.runtime_lib_dir()));
        for dir in &kernel.include_dirs {
            cmd.arg("-I").arg(absolute(dir)?);
        }
        // IRON's rule puts this ahead of the kernel's own flags.
        cmd.arg("-Wno-missing-template-arg-list-after-template-kw");
        cmd.args(&kernel.flags);

        run(cmd, "Peano clang++", &source)?;
        expect_file(&output, "Peano clang++")
    }

    /// Compiles a design's MLIR to `xclbin` + `insts` (the instruction
    /// stream), linking the kernel objects in `design.link` — each is put
    /// into the work directory, where aiecc resolves a core function's
    /// `link_with = "mul.o"`. aiecc packages the xclbin with XRT's
    /// `xclbinutil`, which has to be on its `PATH` (XRT's `setup.sh` puts it
    /// there).
    pub fn compile_design(&self, design: &Design, xclbin: &Path, insts: &Path) -> Result<(), Error> {
        let mlir_path = absolute(&design.mlir)?;
        let mlir = std::fs::read_to_string(&mlir_path)
            .map_err(|e| Error::Compile(format!("reading {}: {e}", mlir_path.display())))?;
        let work_dir = absolute(&design.work_dir)?;
        let xclbin = absolute(xclbin)?;
        let insts = absolute(insts)?;
        create_dir(&work_dir)?;
        for out in [&xclbin, &insts] {
            if let Some(dir) = out.parent() {
                create_dir(dir)?;
            }
            // Success is judged by the outputs appearing, and they live
            // outside the work directory: a stale one from an earlier build
            // would make a failed compile look like it worked, and the
            // caller then runs the previous kernel.
            remove_stale(out)?;
        }

        // The MLIR goes into the work directory as `aie.mlir`, next to the
        // objects it links, as mlir-aie's compile_mlir_module lays it out.
        let input = work_dir.join("aie.mlir");
        std::fs::write(&input, &mlir).map_err(|e| Error::Compile(format!("writing {}: {e}", input.display())))?;
        for obj in &design.link {
            let obj = absolute(obj)?;
            let name = obj.file_name().ok_or_else(|| Error::Compile(format!("{} names no file", obj.display())))?;
            let dst = work_dir.join(name);
            if dst != obj {
                std::fs::copy(&obj, &dst)
                    .map_err(|e| Error::Compile(format!("copying {} to {}: {e}", obj.display(), dst.display())))?;
            }
        }

        let mut cmd = self.command(&self.aiecc);
        cmd.arg(&input);
        cmd.arg(format!("--peano={}", self.peano.display()));
        cmd.arg("--get-npu-insts").arg(format!("--npu-insts-name={}", insts.display()));
        cmd.arg("--get-xclbin").arg(format!("--xclbin-name={}", xclbin.display()));
        cmd.arg(format!("--tmpdir={}", work_dir.display()));
        cmd.arg(format!("-j{}", design.jobs));
        cmd.args(&design.flags);
        cmd.arg(format!("--xclbin-kernel-name={}", design.kernel_name));
        cmd.current_dir(&work_dir);

        run(cmd, "aiecc", &mlir_path)?;
        expect_file(&xclbin, "aiecc")?;
        expect_file(&insts, "aiecc")
    }

    fn command(&self, tool: &Path) -> Command {
        match self.launcher.split_first() {
            Some((first, rest)) => {
                let mut cmd = Command::new(first);
                cmd.args(rest).arg(tool);
                cmd
            }
            None => Command::new(tool),
        }
    }
}

/// A kernel source for [`Toolchain::compile_kernel`].
#[derive(Debug, Clone)]
pub struct KernelSource {
    source: PathBuf,
    arch: Arch,
    include_dirs: Vec<PathBuf>,
    flags: Vec<OsString>,
}

impl KernelSource {
    pub fn new(source: impl Into<PathBuf>, arch: Arch) -> Self {
        Self { source: source.into(), arch, include_dirs: Vec::new(), flags: Vec::new() }
    }

    /// Adds `-I dir`.
    pub fn include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.include_dirs.push(dir.into());
        self
    }

    /// Adds `-Dname=value` — how operators size a kernel (`-DDIM_M=64`).
    pub fn define(mut self, name: &str, value: impl fmt::Display) -> Self {
        self.flags.push(format!("-D{name}={value}").into());
        self
    }

    /// Adds a compiler flag as is (an operator's `extra_flags`).
    pub fn flag(mut self, flag: impl Into<OsString>) -> Self {
        self.flags.push(flag.into());
        self
    }
}

/// A design for [`Toolchain::compile_design`].
#[derive(Debug, Clone)]
pub struct Design {
    mlir: PathBuf,
    work_dir: PathBuf,
    link: Vec<PathBuf>,
    kernel_name: String,
    jobs: usize,
    flags: Vec<OsString>,
}

impl Design {
    /// The design in the MLIR file `mlir`, built in `work_dir` (aiecc's
    /// `--tmpdir`; one per design, as its intermediates have fixed names).
    pub fn new(mlir: impl Into<PathBuf>, work_dir: impl Into<PathBuf>) -> Self {
        Self {
            mlir: mlir.into(),
            work_dir: work_dir.into(),
            link: Vec::new(),
            kernel_name: "MLIR_AIE".into(),
            jobs: 1,
            flags: Vec::new(),
        }
    }

    /// Links the kernel object `obj` (a core function's `link_with`).
    pub fn link(mut self, obj: impl Into<PathBuf>) -> Self {
        self.link.push(obj.into());
        self
    }

    /// The kernel's name inside the xclbin (default `MLIR_AIE`, IRON's).
    pub fn kernel_name(mut self, name: impl Into<String>) -> Self {
        self.kernel_name = name.into();
        self
    }

    /// aiecc's parallelism, `-j` (default 1, IRON's `AIECC_JOBS` default).
    pub fn jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs.max(1);
        self
    }

    /// Adds an aiecc flag as is (an artifact's `extra_flags`).
    pub fn flag(mut self, flag: impl Into<OsString>) -> Self {
        self.flags.push(flag.into());
        self
    }
}

fn absolute(p: &Path) -> Result<PathBuf, Error> {
    std::path::absolute(p).map_err(|e| Error::Compile(format!("{}: {e}", p.display())))
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn create_dir(dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(dir).map_err(|e| Error::Compile(format!("creating {}: {e}", dir.display())))
}

fn remove_stale(p: &Path) -> Result<(), Error> {
    match std::fs::remove_file(p) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(Error::Compile(format!("removing stale {}: {e}", p.display())))
        }
        _ => Ok(()),
    }
}

fn expect_file(p: &Path, tool: &str) -> Result<(), Error> {
    if p.is_file() {
        Ok(())
    } else {
        Err(Error::Compile(format!("{tool} reported success but did not write {}", p.display())))
    }
}

/// Runs `cmd`, turning a failure into an error carrying the tool's output.
fn run(mut cmd: Command, tool: &str, input: &Path) -> Result<Output, Error> {
    let out = cmd
        .output()
        .map_err(|e| Error::Compile(format!("starting {tool} ({}): {e}", cmd.get_program().to_string_lossy())))?;
    if out.status.success() {
        return Ok(out);
    }
    // aiecc prints its diagnostics on either stream; keep the end of each,
    // which is where the error is, and of a line only what is left after
    // its progress counter's carriage returns.
    let tail = |b: &[u8]| {
        let s = String::from_utf8_lossy(b);
        let lines: Vec<&str> = s.lines().map(|l| l.rsplit('\r').next().unwrap_or(l).trim_end()).collect();
        lines[lines.len().saturating_sub(40)..].join("\n")
    };
    Err(Error::Compile(format!(
        "{tool} failed ({}) on {}:\n{}\n{}",
        out.status,
        input.display(),
        tail(&out.stderr),
        tail(&out.stdout)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_follows_the_mlir_device() {
        assert_eq!(Arch::of_mlir("module {\n  aie.device(npu2) {"), Some(Arch::Aie2p));
        assert_eq!(Arch::of_mlir("aie.device(npu2_4col) {"), Some(Arch::Aie2p));
        assert_eq!(Arch::of_mlir("aie.device(npu1_1col) {"), Some(Arch::Aie2));
        assert_eq!(Arch::of_mlir("aie.device(xcve2802) {"), None);
        assert_eq!(Arch::of_mlir("module {}"), None);
    }

    #[test]
    fn a_launcher_prefixes_the_tool() {
        let tc = Toolchain::new("/opt/mlir_aie").with_launcher(["ctr", "--quiet"]);
        let cmd = tc.command(Path::new("/opt/mlir_aie/bin/aiecc"));
        assert_eq!(cmd.get_program(), "ctr");
        let args: Vec<_> = cmd.get_args().collect();
        assert_eq!(args, ["--quiet", "/opt/mlir_aie/bin/aiecc"]);
    }

    #[test]
    fn a_missing_kernel_source_is_an_error_not_a_spawn() {
        let tc = Toolchain::new("/nonexistent/mlir_aie");
        let out = std::env::temp_dir().join("iron-xrt-compile-test.o");
        let err = tc.compile_kernel(&KernelSource::new("/nonexistent/k.cc", Arch::Aie2p), &out).unwrap_err();
        assert!(matches!(err, Error::Compile(m) if m.contains("does not exist")));
    }
}
