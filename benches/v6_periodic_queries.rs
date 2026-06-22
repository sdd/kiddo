use codspeed_criterion_compat::{black_box, criterion_group, criterion_main, Criterion};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::Eytzinger;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroUsize;

const K: usize = 3;
const B: usize = 32;
const BOX_SIZE: [f64; K] = [1.0; K];
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_LOG2_POINTS: u32 = 18;
const DEFAULT_MAX_QTY: usize = 16;
const DEFAULT_MAX_DIST: f64 = 0.01;
const POINT_SEED: u64 = 0x5eed_0000_0000_2001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_2002;

type BaselineLeaves = VecOfArenas<f64, u32, K, B>;
type BaselineTree = KdTree<f64, u32, Eytzinger, BaselineLeaves, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_u32_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn read_f64_env(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn build_points(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_periodic_queries(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count)
        .map(|idx| {
            let mut query = rng.random::<[f64; K]>();
            for coord in &mut query {
                let edge_sample = rng.random::<f64>() * 0.05;
                *coord = if idx & 1 == 0 {
                    edge_sample
                } else {
                    1.0 - edge_sample
                };
            }
            query
        })
        .collect()
}

fn run_periodic_nearest_one(tree: &BaselineTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .periodic_boundary_condition(&BOX_SIZE)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_periodic_nearest_n_within(
    tree: &BaselineTree,
    queries: &[[f64; K]],
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .periodic_boundary_condition(&BOX_SIZE)
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .execute();
        checksum_len += results.len();

        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn run_periodic_within(
    tree: &BaselineTree,
    queries: &[[f64; K]],
    max_dist: f64,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .periodic_boundary_condition(&BOX_SIZE)
            .within::<SquaredEuclidean<f64>>(max_dist)
            .execute();
        checksum_len += results.len();

        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn periodic_queries(c: &mut Criterion) {
    let log2_points = read_u32_env("KIDDO_BENCH_LOG2_POINTS", DEFAULT_LOG2_POINTS);
    let point_count = 1usize << log2_points;
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let max_qty = NonZeroUsize::new(read_usize_env("KIDDO_BENCH_MAX_QTY", DEFAULT_MAX_QTY))
        .expect("KIDDO_BENCH_MAX_QTY must be non-zero");
    let max_dist = read_f64_env("KIDDO_BENCH_MAX_DIST", DEFAULT_MAX_DIST);

    let points = build_points(point_count);
    let queries = build_periodic_queries(query_count);
    let tree =
        BaselineTree::new_from_slice(&points).expect("failed to build periodic benchmark tree");

    let mut group = c.benchmark_group("v6 periodic queries");
    group.bench_function("periodic_nearest_one", |b| {
        b.iter(|| black_box(run_periodic_nearest_one(&tree, &queries)))
    });
    group.bench_function("periodic_nearest_n_within", |b| {
        b.iter(|| {
            black_box(run_periodic_nearest_n_within(
                &tree, &queries, max_dist, max_qty,
            ))
        })
    });
    group.bench_function("periodic_within", |b| {
        b.iter(|| black_box(run_periodic_within(&tree, &queries, max_dist)))
    });
    group.finish();
}

criterion_group!(benches, periodic_queries);
criterion_main!(benches);
