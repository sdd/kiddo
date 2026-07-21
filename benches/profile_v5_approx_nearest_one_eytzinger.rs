use codspeed_criterion_compat::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;
const TREE_SIZES: [usize; 11] = [
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
    1 << 25,
    1 << 26,
];

type F64Tree = ImmutableKdTree<f64, u32, K, B>;
type F32Tree = ImmutableKdTree<f32, u32, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn run_queries_f64(tree: &F64Tree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree.approx_nearest_one::<SquaredEuclidean>(black_box(query));
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_queries_f32(tree: &F32Tree, queries: &[[f32; K]]) -> (f32, u64) {
    let mut checksum_dist = 0.0f32;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree.approx_nearest_one::<SquaredEuclidean>(black_box(query));
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn approx_nearest_one(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);

    eprintln!(
        "benchmarking v5 approx_nearest_one: dims={} tree_sizes=2^16..2^26 queries={} point_seed={} query_seed={}",
        K, query_count, POINT_SEED, QUERY_SEED
    );

    let f64_queries = build_queries_f64(query_count);
    let mut f64_group = c.benchmark_group("profile_v5_approx_nearest_one_eytzinger/f64");
    f64_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in TREE_SIZES {
        let points = build_points_f64(point_count);
        let tree: F64Tree = ImmutableKdTree::new_from_slice(&points);
        f64_group.bench_function(BenchmarkId::new("approx_nearest_one", point_count), |b| {
            b.iter(|| black_box(run_queries_f64(&tree, &f64_queries)));
        });
    }
    f64_group.finish();

    let f32_queries = build_queries_f32(query_count);
    let mut f32_group = c.benchmark_group("profile_v5_approx_nearest_one_eytzinger/f32");
    f32_group.throughput(Throughput::Elements(query_count as u64));
    for point_count in TREE_SIZES {
        let points = build_points_f32(point_count);
        let tree: F32Tree = ImmutableKdTree::new_from_slice(&points);
        f32_group.bench_function(BenchmarkId::new("approx_nearest_one", point_count), |b| {
            b.iter(|| black_box(run_queries_f32(&tree, &f32_queries)));
        });
    }
    f32_group.finish();
}

criterion_group!(benches, approx_nearest_one);
criterion_main!(benches);
