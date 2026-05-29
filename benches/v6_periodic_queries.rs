use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroUsize;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const DEFAULT_POINT_COUNT: usize = 1usize << 22;
const DEFAULT_MAX_QTY: usize = 16;
const DEFAULT_MAX_DIST: f64 = 0.0025;
const POINT_SEED: u64 = 0x5eed_0000_0000_0401;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0402;
const BOX_SIZE: [f64; K] = [1.0; K];

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type DonnellyPfTree = KdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
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

fn build_queries(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn run_nearest_one_queries(tree: &DonnellyPfTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_periodic_nearest_one_queries(tree: &DonnellyPfTree, queries: &[[f64; K]]) -> (f64, u64) {
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

fn run_periodic_nearest_n_within_queries(
    tree: &DonnellyPfTree,
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

fn v6_periodic_queries(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let point_count = read_usize_env("KIDDO_BENCH_POINTS", DEFAULT_POINT_COUNT);
    let max_qty =
        NonZeroUsize::new(read_usize_env("KIDDO_BENCH_MAX_QTY", DEFAULT_MAX_QTY)).unwrap();
    let max_dist = read_f64_env("KIDDO_BENCH_MAX_DIST", DEFAULT_MAX_DIST);

    let points = build_points(point_count);
    let queries = build_queries(query_count);
    let tree: DonnellyPfTree = KdTree::new_from_slice(&points).unwrap();

    let mut group = c.benchmark_group("v6 periodic queries");
    group.throughput(Throughput::Elements(query_count as u64));

    group.bench_function(
        BenchmarkId::new("nearest_one / Donnelly PF", point_count),
        |b| {
            b.iter(|| black_box(run_nearest_one_queries(&tree, &queries)));
        },
    );

    group.bench_function(
        BenchmarkId::new("nearest_one periodic / Donnelly PF", point_count),
        |b| {
            b.iter(|| black_box(run_periodic_nearest_one_queries(&tree, &queries)));
        },
    );

    group.bench_function(
        BenchmarkId::new("nearest_n_within periodic / Donnelly PF", point_count),
        |b| {
            b.iter(|| {
                black_box(run_periodic_nearest_n_within_queries(
                    &tree, &queries, max_dist, max_qty,
                ))
            });
        },
    );

    group.finish();
}

criterion_group!(benches, v6_periodic_queries);
criterion_main!(benches);
