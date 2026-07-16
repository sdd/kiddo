use std::env;
use std::io;
use std::path::Path;
use std::process::Command;

fn main() -> io::Result<()> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/criterion-export/Cargo.toml");
    let status = Command::new("cargo")
        .arg("run")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--")
        .args(env::args_os().skip(1))
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "Criterion exporter exited with {status}"
        )));
    }

    Ok(())
}
