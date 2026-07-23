#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use std::hint::black_box;
use std::time::Instant;

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use kiddo::{QueryResultItem, SquaredEuclidean};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_TREE_SIZE: usize = 1 << 20;
const DEFAULT_QUERY_COUNT: usize = 256;
const DEFAULT_SAMPLE_COUNT: usize = 5;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;

// Physical radius, then a deliberately rounded result-count estimate.
const CASES: [(f64, usize); 3] = [(0.03, 128), (0.06, 1_024), (0.1, 4_608)];

type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Checksum {
    len: usize,
    item_sum: u64,
    distance_bits_sum: u64,
}

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[inline(always)]
fn execute_query(
    tree: &Tree,
    query: &[f64; K],
    max_dist: f64,
    sorted: bool,
    capacity: Option<usize>,
) -> Vec<QueryResultItem<(), u32, f64>> {
    match (sorted, capacity) {
        (true, Some(capacity)) => tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(max_dist)
            .with_result_capacity(capacity)
            .execute(),
        (true, None) => tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(max_dist)
            .execute(),
        (false, Some(capacity)) => tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(max_dist)
            .unsorted()
            .with_result_capacity(capacity)
            .execute(),
        (false, None) => tree
            .query(black_box(query))
            .within::<SquaredEuclidean<f64>>(max_dist)
            .unsorted()
            .execute(),
    }
}

fn execute_queries(
    tree: &Tree,
    queries: &[[f64; K]],
    max_dist: f64,
    sorted: bool,
    capacity: Option<usize>,
) -> Checksum {
    let mut checksum = Checksum {
        len: 0,
        item_sum: 0,
        distance_bits_sum: 0,
    };

    for query in queries {
        let results = execute_query(tree, query, max_dist, sorted, capacity);

        checksum.len = checksum.len.wrapping_add(results.len());
        for result in results {
            checksum.item_sum = checksum.item_sum.wrapping_add(result.item as u64);
            checksum.distance_bits_sum = checksum
                .distance_bits_sum
                .wrapping_add(result.distance.to_bits());
        }
    }

    black_box(checksum)
}

fn measure(
    tree: &Tree,
    queries: &[[f64; K]],
    max_dist: f64,
    sorted: bool,
    capacity: Option<usize>,
) -> (f64, usize) {
    // Result contents are validated outside the timed samples. Keep normal
    // container destruction here, but do not include result iteration.
    let start = Instant::now();
    let mut total_len = 0usize;
    for query in queries {
        let results = execute_query(tree, query, max_dist, sorted, capacity);
        total_len = total_len.wrapping_add(results.len());
        let _ = black_box(results);
    }
    (
        start.elapsed().as_nanos() as f64 / queries.len() as f64,
        black_box(total_len),
    )
}

fn main() {
    let tree_size = read_usize_env("KIDDO_PROFILE_TREE_SIZE", DEFAULT_TREE_SIZE);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let sample_count = read_usize_env("KIDDO_PROFILE_SAMPLES", DEFAULT_SAMPLE_COUNT);
    assert!(tree_size > 0 && query_count > 0 && sample_count > 0);

    let mut point_rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    let points: Vec<_> = (0..tree_size)
        .map(|_| point_rng.random::<[f64; K]>())
        .collect();
    let tree = Tree::new_from_slice(&points).unwrap();

    let mut query_rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    let queries: Vec<_> = (0..query_count)
        .map(|_| query_rng.random::<[f64; K]>())
        .collect();

    println!(
        "# result capacity profile: arch={} tree_size={} queries={} samples={}",
        std::env::consts::ARCH,
        tree_size,
        query_count,
        sample_count
    );
    println!("case_header,mode,radius,capacity,avg_results,default_ns,hinted_ns,speedup_pct");

    for sorted in [true, false] {
        for (radius, default_capacity) in CASES {
            let capacity = default_capacity
                .saturating_mul(tree_size)
                .div_ceil(DEFAULT_TREE_SIZE)
                .max(1);
            let max_dist = radius * radius;
            let expected = execute_queries(&tree, &queries, max_dist, sorted, None);
            assert_eq!(
                execute_queries(&tree, &queries, max_dist, sorted, Some(capacity)),
                expected
            );

            let mut default_samples = Vec::with_capacity(sample_count);
            let mut hinted_samples = Vec::with_capacity(sample_count);
            for sample_idx in 0..sample_count {
                let capacities = if sample_idx.is_multiple_of(2) {
                    [None, Some(capacity)]
                } else {
                    [Some(capacity), None]
                };
                for current_capacity in capacities {
                    let (elapsed, total_len) =
                        measure(&tree, &queries, max_dist, sorted, current_capacity);
                    assert_eq!(total_len, expected.len);
                    if current_capacity.is_some() {
                        hinted_samples.push(elapsed);
                    } else {
                        default_samples.push(elapsed);
                    }
                }
            }

            let default_ns = median(default_samples);
            let hinted_ns = median(hinted_samples);
            let speedup = (default_ns - hinted_ns) / default_ns * 100.0;
            println!(
                "case,{},{radius:.3},{capacity},{:.1},{default_ns:.2},{hinted_ns:.2},{speedup:.2}",
                if sorted { "sorted" } else { "unsorted" },
                expected.len as f64 / queries.len() as f64,
            );
        }
    }
}
