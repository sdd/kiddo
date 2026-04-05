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
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=2" \
    cargo rustc --lib --release -- --emit asm -o target/donnelly_get_idx_v2_x86_64_v4.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_x86_64_v4.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"

# Generate Apple M2 assembly for donnelly_get_idx_v2
asm-m4:
    RUSTFLAGS="-C target-cpu=apple-m4 -C opt-level=2" \
    cargo rustc --lib --release --features no_inline -- --emit asm -o target/donnelly_get_idx_v2_apple_m2.s
    @echo "Assembly output written to target/donnelly_get_idx_v2_apple_m2.s"
    @echo "Search for 'donnelly_get_idx_v2' in the file to find the function"


asm-k6-nearest-one-eytz:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "kiddo::immutable::float::query::nearest_one::cargo_asm::v6_nearest_one_eytzinger_with_stack" > v6_nearest_one_eytzinger.asm

asm-k6-nearest-one-eytz-v3:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3.asm

asm-k6-nearest-one-eytz-v3-core:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core.asm

asm-k6-nearest-one-eytz-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_eytzinger_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_cargo_asm_hook" > v6_approx_nearest_one_eytzinger_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_v3_avx512_clean.asm

asm-k6-approx-nearest-one-eytz-voa-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arrays_cargo_asm_hook" > v6_approx_nearest_one_eytzinger_vec_of_arrays_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-voa-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arrays_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_vec_of_arrays_v3_avx512_clean.asm

asm-k6-approx-nearest-one-eytz-voarena-v3-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arenas_cargo_asm_hook" > v6_approx_nearest_one_eytzinger_vec_of_arenas_v3_avx512.asm

asm-k6-approx-nearest-one-eytz-voarena-v3-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_approx_nearest_one_eytzinger_vec_of_arenas_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_approx_nearest_one_eytzinger_vec_of_arenas_v3_avx512_clean.asm

asm-k6-nearest-one-arena-fallback-v3:
    cargo asm --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_with_query_wide_arena_fallback_cargo_asm_hook" > v6_nearest_one_with_query_wide_arena_fallback_v3.asm

asm-k6-nearest-one-arena-fallback-v3-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" "v6_nearest_one_with_query_wide_arena_fallback_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_with_query_wide_arena_fallback_v3_clean.asm

asm-k6-nearest-one-donnelly-block3-fill-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_fill_backtrack_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_fill_avx512_clean.asm

asm-k6-nearest-one-donnelly-block3-pending-select-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "donnelly_block3_pending_select_f64_cargo_asm_hook" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_donnelly_block3_pending_select_avx512_clean.asm

asm-k6-nearest-one-eytz-v3-core-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook" > v6_nearest_one_eytzinger_v3_core_avx512.asm

asm-k6-nearest-one-eytz-v3-leaf-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::kd_tree::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_chunked_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" > v6_nearest_one_eytzinger_v3_leaf_avx512.asm

asm-k6-nearest-one-eytz-v3-leaf-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::kd_tree::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_chunked_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_eytzinger_v3_leaf_avx512_clean.asm

asm-k6-nearest-one-voarena-v3-leaf-avx512:
    cargo asm --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::kd_tree::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_arena_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" > v6_nearest_one_arena_leaf_k3_avx512.asm

asm-k6-nearest-one-voarena-v3-leaf-avx512-clean:
    RUSTC_WRAPPER= cargo asm --simplify --features simd,cargo_asm,logging_off --lib --target-cpu=native -C="opt-level=2" -C="target-cpu=native" "kiddo::kd_tree::leaf_view_chunked::nearest_one::avx512::leaf_nearest_one_arena_nozero_f64_k3::<f64, kiddo::dist::squared_euclidean::SquaredEuclidean<f64>, usize>" | python3 scripts/clean_cargo_asm.py > v6_nearest_one_arena_leaf_k3_avx512_clean.asm

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

build-profile-v6-leaf-strategies FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies

perf-v6-leaf-strategies QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat -d -d -d \
        -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses \
        ./target/release/profile_v6_leaf_strategies

perf-v6-leaf-strategies-branch QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e cycles,instructions,branches,branch-misses \
        ./target/release/profile_v6_leaf_strategies

perf-v6-leaf-strategies-cache QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e cache-misses,L1-dcache-load-misses,l2_cache_req_stat.ls_rd_blk_c,ls_dmnd_fills_from_sys.local_l2,ls_dmnd_fills_from_sys.local_ccx,ls_dmnd_fills_from_sys.dram_io_near \
        ./target/release/profile_v6_leaf_strategies

perf-v6-leaf-strategies-other QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e dTLB-loads,dTLB-load-misses,itlb-loads,itlb-load-misses,ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit,ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss \
        ./target/release/profile_v6_leaf_strategies

perf-v6-leaf-strategies-prefetch QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    perf stat \
        -e ls_pref_instr_disp.prefetch_nta,ls_inef_sw_pref.mab_mch_cnt,ls_sw_pf_dc_fills.local_ccx,l2_pf_hit_l2.l2_hwpf,l2_pf_miss_l2_hit_l3.l2_hwpf,l2_pf_miss_l2_l3.l2_hwpf \
        ./target/release/profile_v6_leaf_strategies

uprof-v6-leaf-strategies QUERY='nearest' STRATEGY='arena' POINTS='4194304' QUERIES='1000' REPEATS='2000' FEATURES='simd,test_utils' OUT='./uprof-output-v6-leaf-strategies':
    cargo build --release --features {{FEATURES}} --bin profile_v6_leaf_strategies
    KIDDO_PROFILE_QUERY_KIND={{QUERY}} \
    KIDDO_PROFILE_STRATEGY={{STRATEGY}} \
    KIDDO_PROFILE_POINTS={{POINTS}} \
    KIDDO_PROFILE_QUERIES={{QUERIES}} \
    KIDDO_PROFILE_QUERY_BATCH_REPEATS={{REPEATS}} \
    /opt/AMD/AMDuProf_Linux_x64_5.1.701/bin/AMDuProfCLI collect \
        --config ibs \
        --interval 10000 \
        --format csv \
        -w /home/scotty/projects/kiddo \
        -o {{OUT}} \
        ./target/release/profile_v6_leaf_strategies

build-profile-v5-nearest-one-eytz:
    cargo build --release --features profile_v5 --bin profile_v5_nearest_one_eytzinger

profile-v5-nearest-one-eytz-samply: build-profile-v5-nearest-one-eytz
    KIDDO_PROFILE_QUERY_BATCH_REPEATS=2000 samply record ./target/release/profile_v5_nearest_one_eytzinger


build:
    RUSTFLAGS="-C target-cpu=znver3 -C opt-level=2" cargo build --release --example immutable-large-ann-donnelly --example immutable-large-ann-eytzinger

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
