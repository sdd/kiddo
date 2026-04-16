use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::leaf_strategies::VecOfArenas;
use kiddo::kd_tree::KdTree;
use kiddo::stem_strategies::donnelly_2_pf::DonnellyPf;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroUsize;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_00;
const DEFAULT_POINT_COUNT: usize = 1usize << 24;
const DEFAULT_MAX_QTY: usize = 16;
const DEFAULT_MAX_DIST: f64 = 0.0025;
const POINT_SEED: u64 = 0x5eed_0000_0000_0301;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0302;

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

fn run_sorted_nearest_n_within_queries(
    tree: &DonnellyPfTree,
    queries: &[[f64; K]],
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree.nearest_n_within::<SquaredEuclidean<f64>>(
            black_box(query),
            max_dist,
            max_qty,
            true,
        );
        checksum_len += results.len();

        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn run_best_n_within_queries(
    tree: &DonnellyPfTree,
    queries: &[[f64; K]],
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results =
            tree.best_n_within::<SquaredEuclidean<f64>>(black_box(query), max_dist, max_qty);
        checksum_len += results.len();

        for result in results.into_vec() {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn v6_result_collection_focus(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let point_count = read_usize_env("KIDDO_BENCH_POINTS", DEFAULT_POINT_COUNT);
    let max_qty =
        NonZeroUsize::new(read_usize_env("KIDDO_BENCH_MAX_QTY", DEFAULT_MAX_QTY)).unwrap();
    let max_dist = read_f64_env("KIDDO_BENCH_MAX_DIST", DEFAULT_MAX_DIST);

    let points = build_points(point_count);
    let queries = build_queries(query_count);
    let tree: DonnellyPfTree = KdTree::new_from_slice(&points);

    let mut group = c.benchmark_group("v6 result collection focus");
    group.throughput(Throughput::Elements(query_count as u64));

    group.bench_function(
        BenchmarkId::new("sorted nearest_n_within / Donnelly PF", point_count),
        |b| {
            b.iter(|| {
                black_box(run_sorted_nearest_n_within_queries(
                    &tree, &queries, max_dist, max_qty,
                ))
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("best_n_within / Donnelly PF", point_count),
        |b| {
            b.iter(|| {
                black_box(run_best_n_within_queries(
                    &tree, &queries, max_dist, max_qty,
                ))
            });
        },
    );

    group.finish();
}

criterion_group!(benches, v6_result_collection_focus);
criterion_main!(benches);
