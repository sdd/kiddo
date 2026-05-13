use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
use kiddo::stem_strategy::eytzinger_pf_far::EytzingerPfFar;
use kiddo::stem_strategy::Eytzinger;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
use kiddo::stem_strategy::{Block3, DonnellyMarkerSimd};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const DEFAULT_POINT_COUNT: usize = 1usize << 22;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type EytzingerTree = KdTree<f64, u32, Eytzinger<K>, ArenaLeaves, K, B>;
type EytzingerPfFarTree = KdTree<f64, u32, EytzingerPfFar<K, 8>, ArenaLeaves, K, B>;
type DonnellyPfTree = KdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
type DonnellySimdTree = KdTree<f64, u32, DonnellyMarkerSimd<Block3, 64, 8, K>, ArenaLeaves, K, B>;

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

fn run_nearest_queries_eytzinger(tree: &EytzingerTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_nearest_queries_eytzinger_pf_far(
    tree: &EytzingerPfFarTree,
    queries: &[[f64; K]],
) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_nearest_queries_donnelly(tree: &DonnellyPfTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
fn run_nearest_queries_donnelly_simd(tree: &DonnellySimdTree, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
        checksum_dist += dist;
        checksum_item = checksum_item.wrapping_add(item as u64);
    }

    (checksum_dist, checksum_item)
}

fn v6_stem_strategies_focus(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let point_count = read_usize_env("KIDDO_BENCH_POINTS", DEFAULT_POINT_COUNT);
    let points = build_points(point_count);
    let queries = build_queries(query_count);

    let eytzinger_tree: EytzingerTree = KdTree::new_from_slice(&points).unwrap();
    let eytzinger_pf_far_tree: EytzingerPfFarTree = KdTree::new_from_slice(&points).unwrap();
    let donnelly_tree: DonnellyPfTree = KdTree::new_from_slice(&points).unwrap();
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    let donnelly_simd_tree: DonnellySimdTree = KdTree::new_from_slice(&points).unwrap();

    let mut group = c.benchmark_group("v6 nearest_one stem strategies focus");
    group.throughput(Throughput::Elements(query_count as u64));

    group.bench_function(BenchmarkId::new("Eytzinger", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_eytzinger(&eytzinger_tree, &queries)));
    });

    group.bench_function(BenchmarkId::new("Eytzinger PF Far", point_count), |b| {
        b.iter(|| {
            black_box(run_nearest_queries_eytzinger_pf_far(
                &eytzinger_pf_far_tree,
                &queries,
            ))
        });
    });

    group.bench_function(BenchmarkId::new("Donnelly PF", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_donnelly(&donnelly_tree, &queries)));
    });

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    group.bench_function(BenchmarkId::new("Donnelly Block SIMD", point_count), |b| {
        b.iter(|| {
            black_box(run_nearest_queries_donnelly_simd(
                &donnelly_simd_tree,
                &queries,
            ))
        });
    });

    group.finish();
}

criterion_group!(benches, v6_stem_strategies_focus);
criterion_main!(benches);
