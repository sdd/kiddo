#!/usr/bin/env just --justfile

default:
  just --list

test-donnelly:
    cargo test donnelly

test-fast ARGS='':
    cargo test --profile fast-tests {{ARGS}}

test-fast-simd ARGS='':
    cargo test --profile fast-tests --features simd {{ARGS}}

test-fast-lib FILTER:
    cargo test --profile fast-tests --lib {{FILTER}}

test-fast-v6-nearest-one-large-f32:
    cargo test --profile fast-tests --lib v6_query_nearest_one_large_f32

fuzz-kd-tree:
    RUST_TEST_THREADS=1 cargo test --release --test kd_tree_fuzz -- --ignored --nocapture

fuzz-kd-tree-v6:
    RUST_TEST_THREADS=1 cargo test --release --test kd_tree_fuzz_v6 -- --ignored --nocapture

fuzz-kd-tree-v6-non-simd:
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=1 KIDDO_FUZZ_V6_RUN_SIMD=0 cargo test --release --test kd_tree_fuzz_v6 -- --ignored --nocapture

fuzz-kd-tree-v6-simd:
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=0 KIDDO_FUZZ_V6_RUN_SIMD=1 cargo test --release --features simd --test kd_tree_fuzz_v6 -- --ignored --nocapture

fuzz-kd-tree-v6-simd-fast:
    RUST_TEST_THREADS=1 KIDDO_FUZZ_V6_RUN_NON_SIMD=0 KIDDO_FUZZ_V6_RUN_SIMD=1 KIDDO_FUZZ_V6_SIMD_FAST=1 cargo test --profile fast-tests --features simd --test kd_tree_fuzz_v6 -- --ignored --nocapture

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


asm-k6-nearest-one-eytz:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=3" "kiddo::immutable::float::query::nearest_one::cargo_asm::v6_nearest_one_eytzinger_with_stack" > v6_nearest_one_eytzinger.asm

asm-k6-nearest-one-eytz-v3:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=3" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3.asm

asm-k6-nearest-one-eytz-v3-core:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=3" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core.asm

asm-k6-nearest-one-eytz-v3-avx512:
    cargo asm --features cargo_asm,logging_off --lib --target x86_64-apple-darwin -C="opt-level=2" -C="target-cpu=skylake-avx512" -C="target-feature=+avx512f,+avx512vl,+avx2,+fma" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_avx512.asm

asm-k6-nearest-one-eytz-v3-core-avx512:
    cargo asm --features cargo_asm,logging_off --lib --target x86_64-apple-darwin -C="opt-level=2" -C="target-cpu=skylake-avx512" -C="target-feature=+avx512f,+avx512vl,+avx2,+fma" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core_avx512.asm

objdump-k6-nearest-one-eytz:
    cargo objdump --release --lib --features cargo_asm,logging_off -- --disassemble-symbols="kiddo::immutable::float::query::nearest_one::cargo_asm::v6_nearest_one_eytzinger_with_stack" --demangle > v6_nearest_one_eytzinger.objdump

objdump-k5-nearest-one:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --disassemble-symbols="kiddo::immutable::float::kdtree::cargo_asm::v5_nearest_one_immutable" --demangle > v5_immutable_nearest_one.objdump

objdump-k5-symbols:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --syms --demangle > v5_symbols.objdump

objdump-k5-nearest-one-recurse SYMBOL:
    cd ../kiddo-v5 && cargo objdump --release --lib -- --disassemble-symbols="{{SYMBOL}}" --demangle > v5_nearest_one_recurse.objdump

objdump-profile-v5-symbols:
    cargo objdump --release --bin profile_v5_nearest_one_eytzinger --features profile_v5 -- --syms --demangle > profile_v5_symbols.objdump

objdump-profile-v5-symbol SYMBOL:
    cargo objdump --release --bin profile_v5_nearest_one_eytzinger --features profile_v5 -- --disassemble-symbols="{{SYMBOL}}" --demangle > profile_v5_symbol.objdump

build-profile-v6-nearest-one-eytz:
    cargo build --release --features test_utils --bin profile_v6_nearest_one_eytzinger

profile-v6-nearest-one-eytz-samply: build-profile-v6-nearest-one-eytz
    KIDDO_PROFILE_QUERY_BATCH_REPEATS=2000 samply record ./target/release/profile_v6_nearest_one_eytzinger

build-profile-v5-nearest-one-eytz:
    cargo build --release --features profile_v5 --bin profile_v5_nearest_one_eytzinger

profile-v5-nearest-one-eytz-samply: build-profile-v5-nearest-one-eytz
    KIDDO_PROFILE_QUERY_BATCH_REPEATS=2000 samply record ./target/release/profile_v5_nearest_one_eytzinger


build:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=3" cargo build --release --example immutable-large-ann-donnelly --example immutable-large-ann-eytzinger

cg-donnelly: build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes \
             --cachegrind-out-file=cachegrind.out.donnelly \
             target/release/examples/immutable-large-ann-donnelly
    cg_annotate cachegrind.out.donnelly > cachegrind.annot.donnelly

cg-eytzinger: build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes \
             --cachegrind-out-file=cachegrind.out.eytzinger \
             target/release/examples/immutable-large-ann-eytzinger
    cg_annotate cachegrind.out.eytzinger > cachegrind.annot.eytzinger

cg-diff: cg-donnelly cg-eytzinger
    cg_diff cachegrind.out.donnelly cachegrind.out.eytzinger \
      | cg_annotate > cachegrind.diff.txt
    @echo "Diff written to cachegrind.diff.txt"

perf-donnelly:
    perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,branch-misses ./target/release/examples/immutable-large-ann-donnelly

perf-eytzinger:
    perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,branch-misses ./target/release/examples/immutable-large-ann-eytzinger


uprof-eytzinger:
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o ./uprof-output-eytz \
        target/release/examples/immutable-large-ann-eytzinger-deserialize-and-query

uprof-donnelly:
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o ./uprof-output-eytz \
        target/release/examples/immutable-large-ann-donnelly-deserialize-and-query
