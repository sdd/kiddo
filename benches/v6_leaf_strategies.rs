use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use kiddo::kd_tree::leaf_strategies::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::kd_tree::KdTree;
use kiddo::traits_unified_2::SquaredEuclidean;
use kiddo::Eytzinger;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const MIN_LOG2_POINTS: u32 = 16;
const MAX_LOG2_POINTS: u32 = 26;
const POINT_SEED: u64 = 0x5eed_0000_0000_0101;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0102;

type FlatTree = KdTree<f64, u32, Eytzinger<K>, FlatVec<f64, u32, K, B>, K, B>;
type ArenaTree = KdTree<f64, u32, Eytzinger<K>, VecOfArenas<f64, u32, K, B>, K, B>;
type VecOfArraysTree = KdTree<f64, u32, Eytzinger<K>, VecOfArrays<f64, u32, K, B>, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
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

fn run_nearest_queries_flat(tree: &FlatTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_nearest_queries_arena(tree: &ArenaTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_nearest_queries_vec_of_arrays(tree: &VecOfArraysTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_approx_queries_flat(tree: &FlatTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.approx_nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_approx_queries_arena(tree: &ArenaTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.approx_nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_approx_queries_vec_of_arrays(tree: &VecOfArraysTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.approx_nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn bench_nearest_group(
    group: &mut BenchmarkGroup<WallTime>,
    point_count: usize,
    queries: &[[f64; K]],
    flat_tree: &FlatTree,
    arena_tree: &ArenaTree,
    vec_of_arrays_tree: &VecOfArraysTree,
) {
    group.bench_function(BenchmarkId::new("FlatVec", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_flat(flat_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("VecOfArenas", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_arena(arena_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("VecOfArrays", point_count), |b| {
        b.iter(|| {
            black_box(run_nearest_queries_vec_of_arrays(
                vec_of_arrays_tree,
                queries,
            ))
        });
    });
}

fn bench_approx_group(
    group: &mut BenchmarkGroup<WallTime>,
    point_count: usize,
    queries: &[[f64; K]],
    flat_tree: &FlatTree,
    arena_tree: &ArenaTree,
    vec_of_arrays_tree: &VecOfArraysTree,
) {
    group.bench_function(BenchmarkId::new("FlatVec", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_flat(flat_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("VecOfArenas", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_arena(arena_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("VecOfArrays", point_count), |b| {
        b.iter(|| {
            black_box(run_approx_queries_vec_of_arrays(
                vec_of_arrays_tree,
                queries,
            ))
        });
    });
}

fn v6_leaf_strategies(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let queries = build_queries(query_count);

    let mut nearest_group = c.benchmark_group("v6 nearest_one leaf strategies");
    nearest_group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in MIN_LOG2_POINTS..=MAX_LOG2_POINTS {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        let flat_tree: FlatTree = KdTree::new_from_slice(&points);
        let arena_tree: ArenaTree = KdTree::new_from_slice(&points);
        let vec_of_arrays_tree: VecOfArraysTree = KdTree::new_from_slice(&points);

        bench_nearest_group(
            &mut nearest_group,
            point_count,
            &queries,
            &flat_tree,
            &arena_tree,
            &vec_of_arrays_tree,
        );
    }
    nearest_group.finish();

    let mut approx_group = c.benchmark_group("v6 approx_nearest_one leaf strategies");
    approx_group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in MIN_LOG2_POINTS..=MAX_LOG2_POINTS {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        let flat_tree: FlatTree = KdTree::new_from_slice(&points);
        let arena_tree: ArenaTree = KdTree::new_from_slice(&points);
        let vec_of_arrays_tree: VecOfArraysTree = KdTree::new_from_slice(&points);

        bench_approx_group(
            &mut approx_group,
            point_count,
            &queries,
            &flat_tree,
            &arena_tree,
            &vec_of_arrays_tree,
        );
    }
    approx_group.finish();
}

criterion_group!(benches, v6_leaf_strategies);
criterion_main!(benches);
