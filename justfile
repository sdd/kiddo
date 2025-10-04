#!/usr/bin/env just --justfile

default:
  just --list

test-donnelly:
    cargo test donnelly

bench-d-v2:
    cargo bench --bench donnelly_v2

bench-d-v2b:
    cargo bench --bench donnelly_v2_branchless

# Generate x86-64-v4 assembly for donnelly_get_idx_v2
asm-x86-v4:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=3" \
    cargo rustc --lib --release -- --emit asm -o target/donnelly_get_idx_v2_x86_64_v4.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_x86_64_v4.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"

# Generate Apple M2 assembly for donnelly_get_idx_v2
asm-m4:
    RUSTFLAGS="-C target-cpu=apple-m4 -C opt-level=3" \
    cargo rustc --lib --release --features no_inline -- --emit asm -o target/donnelly_get_idx_v2_apple_m2.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_apple_m2.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"

build:
    cargo build --release --example cg_donnelly --example cg_eytzinger

cg-donnelly: build
    valgrind --tool=cachegrind --branch-sim=yes \
             --cachegrind-out-file=cachegrind.out.donnelly \
             target/release/cg_donnelly
    cg_annotate cachegrind.out.donnelly > cachegrind.annot.donnelly

cg-eytzinger: build
    valgrind --tool=cachegrind --branch-sim=yes \
             --cachegrind-out-file=cachegrind.out.eytzinger \
             target/release/cg_eytzinger
    cg_annotate cachegrind.out.eytzinger > cachegrind.annot.eytzinger

cg-diff: cg-donnelly cg-eytzinger
    cg_diff cachegrind.out.donnelly cachegrind.out.eytzinger \
      | cg_annotate > cachegrind.diff.txt
    @echo "Diff written to cachegrind.diff.txt"