//! Build script to detect cache line size and set compile-time flags.
//!
//! For cross-compilation to systems with 128-byte cache lines, override with:
//!     RUSTFLAGS="--cfg cache_line_128" cargo build --target <target>

fn main() {
    // Register the custom cfg to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(cache_line_128)");

    let target = std::env::var("TARGET").unwrap();
    let host = std::env::var("HOST").unwrap();

    // Only detect on native builds to avoid cross-compilation issues
    if target == host {
        // Try L1 data cache first (most relevant for our use case)
        match yep_cache_line_size::get_cache_line_size(
            yep_cache_line_size::CacheLevel::L1,
            yep_cache_line_size::CacheType::Data,
        ) {
            Ok(cache_line_size) => {
                println!(
                    "cargo:warning=Detected L1 data cache line size: {} bytes",
                    cache_line_size
                );

                if cache_line_size >= 128 {
                    println!("cargo:rustc-cfg=cache_line_128");
                    println!(
                        "cargo:warning=Enabling f64 Block4 support (128-byte cache lines detected)"
                    );
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to detect cache line size: {}. Assuming 64-byte cache lines.", e);
            }
        }
    } else {
        println!(
            "cargo:warning=Cross-compiling (host: {}, target: {}). \
             Assuming 64-byte cache lines. Override with RUSTFLAGS='--cfg cache_line_128' if needed.",
            host, target
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
}
