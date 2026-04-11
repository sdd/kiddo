use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::leaf_strategies::VecOfArenas;
use kiddo::kd_tree::KdTree;
use kiddo::stem_strategies::donnelly_2_pf::DonnellyPf;
use kiddo::stem_strategies::eytzinger_pf_far::EytzingerPfFar;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
use kiddo::stem_strategies::{Block3, DonnellyMarkerSimd};
use kiddo::stem_strategies::{Donnelly, DonnellySimdDescent, Eytzinger, EytzingerPf};
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

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type EytzingerTree = KdTree<f64, u32, Eytzinger<K>, ArenaLeaves, K, B>;
type EytzingerPfTree = KdTree<f64, u32, EytzingerPf<K, 8>, ArenaLeaves, K, B>;
type EytzingerPfFarTree = KdTree<f64, u32, EytzingerPfFar<K, 8>, ArenaLeaves, K, B>;
type DonnellyTree = KdTree<f64, u32, Donnelly<3, 64, 8, K>, ArenaLeaves, K, B>;
type DonnellyPfTree = KdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;
type DonnellySimdDescentTree = KdTree<f64, u32, DonnellySimdDescent<64, 8, K>, ArenaLeaves, K, B>;
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

macro_rules! impl_query_runners {
    ($nearest_name:ident, $approx_name:ident, $tree_ty:ty) => {
        fn $nearest_name(tree: &$tree_ty, queries: &[[f64; K]]) -> (f64, u64) {
            let mut checksum_dist = 0.0f64;
            let mut checksum_item = 0u64;

            for query in queries {
                let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
                checksum_dist += dist;
                checksum_item = checksum_item.wrapping_add(item as u64);
            }

            (checksum_dist, checksum_item)
        }

        fn $approx_name(tree: &$tree_ty, queries: &[[f64; K]]) -> (f64, u64) {
            let mut checksum_dist = 0.0f64;
            let mut checksum_item = 0u64;

            for query in queries {
                let (dist, item) =
                    tree.approx_nearest_one::<SquaredEuclidean<f64>>(black_box(query));
                checksum_dist += dist;
                checksum_item = checksum_item.wrapping_add(item as u64);
            }

            (checksum_dist, checksum_item)
        }
    };
}

impl_query_runners!(
    run_nearest_queries_eytzinger,
    run_approx_queries_eytzinger,
    EytzingerTree
);
impl_query_runners!(
    run_nearest_queries_eytzinger_pf,
    run_approx_queries_eytzinger_pf,
    EytzingerPfTree
);
impl_query_runners!(
    run_nearest_queries_eytzinger_pf_far,
    run_approx_queries_eytzinger_pf_far,
    EytzingerPfFarTree
);
impl_query_runners!(
    run_nearest_queries_donnelly,
    run_approx_queries_donnelly,
    DonnellyTree
);
impl_query_runners!(
    run_nearest_queries_donnelly_pf,
    run_approx_queries_donnelly_pf,
    DonnellyPfTree
);
impl_query_runners!(
    run_nearest_queries_donnelly_simd_descent,
    run_approx_queries_donnelly_simd_descent,
    DonnellySimdDescentTree
);
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
impl_query_runners!(
    run_nearest_queries_donnelly_simd,
    run_approx_queries_donnelly_simd,
    DonnellySimdTree
);

fn bench_nearest_group(
    group: &mut BenchmarkGroup<WallTime>,
    point_count: usize,
    queries: &[[f64; K]],
    eytzinger_tree: &EytzingerTree,
    eytzinger_pf_tree: &EytzingerPfTree,
    eytzinger_pf_far_tree: &EytzingerPfFarTree,
    donnelly_tree: &DonnellyTree,
    donnelly_pf_tree: &DonnellyPfTree,
    donnelly_simd_descent_tree: &DonnellySimdDescentTree,
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    donnelly_simd_tree: &DonnellySimdTree,
) {
    group.bench_function(BenchmarkId::new("Eytzinger", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_eytzinger(eytzinger_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Eytzinger PF", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_eytzinger_pf(eytzinger_pf_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Eytzinger PF Far", point_count), |b| {
        b.iter(|| {
            black_box(run_nearest_queries_eytzinger_pf_far(
                eytzinger_pf_far_tree,
                queries,
            ))
        });
    });

    group.bench_function(BenchmarkId::new("Donnelly", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_donnelly(donnelly_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Donnelly PF", point_count), |b| {
        b.iter(|| black_box(run_nearest_queries_donnelly_pf(donnelly_pf_tree, queries)));
    });

    group.bench_function(
        BenchmarkId::new("Donnelly SIMD Descent", point_count),
        |b| {
            b.iter(|| {
                black_box(run_nearest_queries_donnelly_simd_descent(
                    donnelly_simd_descent_tree,
                    queries,
                ))
            });
        },
    );

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    group.bench_function(BenchmarkId::new("Donnelly Block SIMD", point_count), |b| {
        b.iter(|| {
            black_box(run_nearest_queries_donnelly_simd(
                donnelly_simd_tree,
                queries,
            ))
        });
    });
}

fn bench_approx_group(
    group: &mut BenchmarkGroup<WallTime>,
    point_count: usize,
    queries: &[[f64; K]],
    eytzinger_tree: &EytzingerTree,
    eytzinger_pf_tree: &EytzingerPfTree,
    eytzinger_pf_far_tree: &EytzingerPfFarTree,
    donnelly_tree: &DonnellyTree,
    donnelly_pf_tree: &DonnellyPfTree,
    donnelly_simd_descent_tree: &DonnellySimdDescentTree,
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    donnelly_simd_tree: &DonnellySimdTree,
) {
    group.bench_function(BenchmarkId::new("Eytzinger", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_eytzinger(eytzinger_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Eytzinger PF", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_eytzinger_pf(eytzinger_pf_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Eytzinger PF Far", point_count), |b| {
        b.iter(|| {
            black_box(run_approx_queries_eytzinger_pf_far(
                eytzinger_pf_far_tree,
                queries,
            ))
        });
    });

    group.bench_function(BenchmarkId::new("Donnelly", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_donnelly(donnelly_tree, queries)));
    });

    group.bench_function(BenchmarkId::new("Donnelly PF", point_count), |b| {
        b.iter(|| black_box(run_approx_queries_donnelly_pf(donnelly_pf_tree, queries)));
    });

    group.bench_function(
        BenchmarkId::new("Donnelly SIMD Descent", point_count),
        |b| {
            b.iter(|| {
                black_box(run_approx_queries_donnelly_simd_descent(
                    donnelly_simd_descent_tree,
                    queries,
                ))
            });
        },
    );

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    group.bench_function(BenchmarkId::new("Donnelly Block SIMD", point_count), |b| {
        b.iter(|| {
            black_box(run_approx_queries_donnelly_simd(
                donnelly_simd_tree,
                queries,
            ))
        });
    });
}

fn v6_stem_strategies(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let queries = build_queries(query_count);

    let mut nearest_group = c.benchmark_group("v6 nearest_one stem strategies");
    nearest_group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in MIN_LOG2_POINTS..=MAX_LOG2_POINTS {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        let eytzinger_tree: EytzingerTree = KdTree::new_from_slice(&points);
        let eytzinger_pf_tree: EytzingerPfTree = KdTree::new_from_slice(&points);
        let eytzinger_pf_far_tree: EytzingerPfFarTree = KdTree::new_from_slice(&points);
        let donnelly_tree: DonnellyTree = KdTree::new_from_slice(&points);
        let donnelly_pf_tree: DonnellyPfTree = KdTree::new_from_slice(&points);
        let donnelly_simd_descent_tree: DonnellySimdDescentTree = KdTree::new_from_slice(&points);
        #[cfg(all(
            feature = "simd",
            target_arch = "x86_64",
            any(target_feature = "avx2", target_feature = "avx512f")
        ))]
        let donnelly_simd_tree: DonnellySimdTree = KdTree::new_from_slice(&points);

        bench_nearest_group(
            &mut nearest_group,
            point_count,
            &queries,
            &eytzinger_tree,
            &eytzinger_pf_tree,
            &eytzinger_pf_far_tree,
            &donnelly_tree,
            &donnelly_pf_tree,
            &donnelly_simd_descent_tree,
            #[cfg(all(
                feature = "simd",
                target_arch = "x86_64",
                any(target_feature = "avx2", target_feature = "avx512f")
            ))]
            &donnelly_simd_tree,
        );
    }
    nearest_group.finish();

    let mut approx_group = c.benchmark_group("v6 approx_nearest_one stem strategies");
    approx_group.throughput(Throughput::Elements(query_count as u64));
    for log2_points in MIN_LOG2_POINTS..=MAX_LOG2_POINTS {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        let eytzinger_tree: EytzingerTree = KdTree::new_from_slice(&points);
        let eytzinger_pf_tree: EytzingerPfTree = KdTree::new_from_slice(&points);
        let eytzinger_pf_far_tree: EytzingerPfFarTree = KdTree::new_from_slice(&points);
        let donnelly_tree: DonnellyTree = KdTree::new_from_slice(&points);
        let donnelly_pf_tree: DonnellyPfTree = KdTree::new_from_slice(&points);
        let donnelly_simd_descent_tree: DonnellySimdDescentTree = KdTree::new_from_slice(&points);
        #[cfg(all(
            feature = "simd",
            target_arch = "x86_64",
            any(target_feature = "avx2", target_feature = "avx512f")
        ))]
        let donnelly_simd_tree: DonnellySimdTree = KdTree::new_from_slice(&points);

        bench_approx_group(
            &mut approx_group,
            point_count,
            &queries,
            &eytzinger_tree,
            &eytzinger_pf_tree,
            &eytzinger_pf_far_tree,
            &donnelly_tree,
            &donnelly_pf_tree,
            &donnelly_simd_descent_tree,
            #[cfg(all(
                feature = "simd",
                target_arch = "x86_64",
                any(target_feature = "avx2", target_feature = "avx512f")
            ))]
            &donnelly_simd_tree,
        );
    }
    approx_group.finish();
}

criterion_group!(benches, v6_stem_strategies);
criterion_main!(benches);
