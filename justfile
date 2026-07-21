#!/usr/bin/env just --justfile

default:
  just --list

benchmark_result_key := `date -u +%Y%m%dT%H%M%SZ`

benchmark-derive-key REF_NAME PYTHON='python3':
    {{quote(PYTHON)}} scripts/benchmark_site.py derive-key --ref-name {{quote(REF_NAME)}} --baseline-ref-name v5.x.x

benchmark-derive-path-key REF_NAME PYTHON='python3':
    {{quote(PYTHON)}} scripts/benchmark_site.py derive-path-key --ref-name {{quote(REF_NAME)}}

bench-v5-basic RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='test_utils' QUERIES='1000':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v5_nearest_n_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v5-nearest_n-eytzinger-${result_key}.json" \
        profile_v5_nearest_n_eytzinger
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v5_nearest_one_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v5-nearest_one-eytzinger-${result_key}.json" \
        profile_v5_nearest_one_eytzinger
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v5_approx_nearest_one_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v5-approx_nearest_one-eytzinger-${result_key}.json" \
        profile_v5_approx_nearest_one_eytzinger

bench-v5-query-family RESULT_KEY=benchmark_result_key OUTPUT_DIR='.' FEATURES='test_utils' QUERIES='1000':
    #!/usr/bin/env bash
    set -euo pipefail
    result_key={{quote(RESULT_KEY)}}
    output_dir={{quote(OUTPUT_DIR)}}
    if [[ ! "$result_key" =~ ^[A-Za-z0-9][A-Za-z0-9._+:-]*$ ]]; then
        echo "RESULT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'" >&2
        exit 2
    fi
    mkdir -p "$output_dir"
    RUSTC_WRAPPER= \
        KIDDO_PROFILE_QUERIES={{quote(QUERIES)}} \
        RUSTFLAGS='-C target-cpu=native' \
        cargo criterion \
            --bench profile_v5_query_family_eytzinger \
            --features {{quote(FEATURES)}}
    cargo run --quiet --manifest-path tools/criterion-export/Cargo.toml -- \
        target/criterion \
        "$output_dir/bench_result-v5-query-family-eytzinger-${result_key}.json" \
        profile_v5_query_family_eytzinger
