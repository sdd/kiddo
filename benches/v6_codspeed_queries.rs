#![allow(clippy::too_many_lines)]

use codspeed_criterion_compat::{black_box, criterion_group, criterion_main, Criterion};
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
use kiddo::stem_strategy::eytzinger_pf_far::EytzingerPfFar;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
use kiddo::stem_strategy::{Block3, DonnellyMarkerSimd};
use kiddo::stem_strategy::{Donnelly, DonnellySimdDescent, Eytzinger, EytzingerPf};
use kiddo::{LeafStrategy, StemStrategy};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroUsize;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const DEFAULT_LOG2_POINTS: u32 = 26;
const DEFAULT_MAX_QTY: usize = 16;
const DEFAULT_MAX_DIST: f64 = 0.0025;
const POINT_SEED_F64: u64 = 0x5eed_0000_0000_1001;
const QUERY_SEED_F64: u64 = 0x5eed_0000_0000_1002;
const POINT_SEED_F32: u64 = 0x5eed_0000_0000_1003;
const QUERY_SEED_F32: u64 = 0x5eed_0000_0000_1004;

type BaselineLeavesF64 = VecOfArenas<f64, u32, K, B>;
type BaselineLeavesF32 = VecOfArenas<f32, u32, K, B>;

type BaselineTreeF64 = KdTree<f64, u32, EytzingerPf<K, 8>, BaselineLeavesF64, K, B>;
type BaselineTreeF32 = KdTree<f32, u32, EytzingerPf<K, 8>, BaselineLeavesF32, K, B>;

type FlatTreeF64 = KdTree<f64, u32, EytzingerPf<K, 8>, FlatVec<f64, u32, K, B>, K, B>;
type VecOfArraysTreeF64 = KdTree<f64, u32, EytzingerPf<K, 8>, VecOfArrays<f64, u32, K, B>, K, B>;

type EytzingerTreeF64 = KdTree<f64, u32, Eytzinger<K>, BaselineLeavesF64, K, B>;
type EytzingerPfFarTreeF64 = KdTree<f64, u32, EytzingerPfFar<K, 8>, BaselineLeavesF64, K, B>;
type DonnellyTreeF64 = KdTree<f64, u32, Donnelly<3, 64, 8, K>, BaselineLeavesF64, K, B>;
type DonnellyPfTreeF64 = KdTree<f64, u32, DonnellyPf<3, 64, 8, K>, BaselineLeavesF64, K, B>;
type DonnellySimdDescentTreeF64 =
    KdTree<f64, u32, DonnellySimdDescent<64, 8, K>, BaselineLeavesF64, K, B>;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
type DonnellyMarkerSimdTreeF64 =
    KdTree<f64, u32, DonnellyMarkerSimd<Block3, 64, 8, K>, BaselineLeavesF64, K, B>;

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

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED_F64);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED_F64);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED_F32);
    (0..point_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn build_queries_f32(query_count: usize) -> Vec<[f32; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED_F32);
    (0..query_count).map(|_| rng.random::<[f32; K]>()).collect()
}

fn bench_name(
    query_type: &str,
    leaf: &str,
    stem: &str,
    float_type: &str,
    log2_points: u32,
) -> String {
    format!("{query_type}: {leaf} + {stem} {float_type}, 2^{log2_points} items")
}

fn run_nearest_one_f64<SS, LS>(
    tree: &KdTree<f64, u32, SS, LS, K, B>,
    queries: &[[f64; K]],
) -> (f64, u64)
where
    SS: StemStrategy,
    LS: LeafStrategy<f64, u32, SS, K, B>,
{
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

fn run_nearest_one_f32<SS, LS>(
    tree: &KdTree<f32, u32, SS, LS, K, B>,
    queries: &[[f32; K]],
) -> (f64, u64)
where
    SS: StemStrategy,
    LS: LeafStrategy<f32, u32, SS, K, B>,
{
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f32>>()
            .execute();
        checksum_dist += result.distance as f64;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_approx_nearest_one_f64(tree: &BaselineTreeF64, queries: &[[f64; K]]) -> (f64, u64) {
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for query in queries {
        let result = tree
            .query(black_box(query))
            .nearest_one::<SquaredEuclidean<f64>>()
            .approx()
            .execute();
        checksum_dist += result.distance;
        checksum_item = checksum_item.wrapping_add(result.item as u64);
    }

    (checksum_dist, checksum_item)
}

fn run_nearest_n_f64(
    tree: &BaselineTreeF64,
    queries: &[[f64; K]],
    max_qty: NonZeroUsize,
) -> (usize, u64, f64) {
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for query in queries {
        let results = tree
            .query(black_box(query))
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .execute();
        checksum_len += results.len();

        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn run_nearest_n_within_f64(
    tree: &BaselineTreeF64,
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

fn run_nearest_n_within_unsorted_f64(
    tree: &BaselineTreeF64,
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
            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
            .within(max_dist)
            .unsorted()
            .execute();
        checksum_len += results.len();

        for result in results {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn run_best_n_within_f64(
    tree: &BaselineTreeF64,
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
            .best_n_within::<SquaredEuclidean<f64>>(max_dist, max_qty)
            .execute();
        checksum_len += results.len();

        for result in results.into_vec() {
            checksum_item = checksum_item.wrapping_add(result.item as u64);
            checksum_dist += result.distance;
        }
    }

    (checksum_len, checksum_item, checksum_dist)
}

fn v6_codspeed_queries(c: &mut Criterion) {
    let query_count = read_usize_env("KIDDO_BENCH_QUERIES", DEFAULT_QUERY_COUNT);
    let log2_points = read_u32_env("KIDDO_BENCH_LOG2_POINTS", DEFAULT_LOG2_POINTS);
    let point_count = 1usize << log2_points;
    let max_qty =
        NonZeroUsize::new(read_usize_env("KIDDO_BENCH_MAX_QTY", DEFAULT_MAX_QTY)).unwrap();
    let max_dist = read_f64_env("KIDDO_BENCH_MAX_DIST", DEFAULT_MAX_DIST);

    let points_f64 = build_points_f64(point_count);
    let queries_f64 = build_queries_f64(query_count);
    let points_f32 = build_points_f32(point_count);
    let queries_f32 = build_queries_f32(query_count);

    let baseline_tree_f64: BaselineTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();
    let baseline_tree_f32: BaselineTreeF32 = KdTree::new_from_slice(&points_f32).unwrap();

    let flat_tree_f64: FlatTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();
    let vec_of_arrays_tree_f64: VecOfArraysTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();

    let eytzinger_tree_f64: EytzingerTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();
    let eytzinger_pf_far_tree_f64: EytzingerPfFarTreeF64 =
        KdTree::new_from_slice(&points_f64).unwrap();
    let donnelly_tree_f64: DonnellyTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();
    let donnelly_pf_tree_f64: DonnellyPfTreeF64 = KdTree::new_from_slice(&points_f64).unwrap();
    let donnelly_simd_descent_tree_f64: DonnellySimdDescentTreeF64 =
        KdTree::new_from_slice(&points_f64).unwrap();
    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    let donnelly_marker_simd_tree_f64: DonnellyMarkerSimdTreeF64 =
        KdTree::new_from_slice(&points_f64).unwrap();

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_one_f64(&baseline_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name("nearest_one", "FlatVec", "Eytzinger PF", "f64", log2_points),
        |b| b.iter(|| black_box(run_nearest_one_f64(&flat_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArrays",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_one_f64(&vec_of_arrays_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Eytzinger",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_one_f64(&eytzinger_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Eytzinger PF Far",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_nearest_one_f64(
                    &eytzinger_pf_far_tree_f64,
                    &queries_f64,
                ))
            })
        },
    );

    c.bench_function(
        &bench_name("nearest_one", "VecOfArenas", "Donnelly", "f64", log2_points),
        |b| b.iter(|| black_box(run_nearest_one_f64(&donnelly_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Donnelly PF",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_one_f64(&donnelly_pf_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Donnelly SIMD Descent",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_nearest_one_f64(
                    &donnelly_simd_descent_tree_f64,
                    &queries_f64,
                ))
            })
        },
    );

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Donnelly Block SIMD",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_nearest_one_f64(
                    &donnelly_marker_simd_tree_f64,
                    &queries_f64,
                ))
            })
        },
    );

    c.bench_function(
        &bench_name(
            "nearest_one",
            "VecOfArenas",
            "Eytzinger PF",
            "f32",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_one_f32(&baseline_tree_f32, &queries_f32))),
    );

    c.bench_function(
        &bench_name(
            "approx_nearest_one",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_approx_nearest_one_f64(&baseline_tree_f64, &queries_f64))),
    );

    c.bench_function(
        &bench_name(
            "nearest_n",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| b.iter(|| black_box(run_nearest_n_f64(&baseline_tree_f64, &queries_f64, max_qty))),
    );

    c.bench_function(
        &bench_name(
            "nearest_n_within",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_nearest_n_within_f64(
                    &baseline_tree_f64,
                    &queries_f64,
                    max_dist,
                    max_qty,
                ))
            })
        },
    );

    c.bench_function(
        &bench_name(
            "nearest_n_within_unsorted",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_nearest_n_within_unsorted_f64(
                    &baseline_tree_f64,
                    &queries_f64,
                    max_dist,
                    max_qty,
                ))
            })
        },
    );

    c.bench_function(
        &bench_name(
            "best_n_within",
            "VecOfArenas",
            "Eytzinger PF",
            "f64",
            log2_points,
        ),
        |b| {
            b.iter(|| {
                black_box(run_best_n_within_f64(
                    &baseline_tree_f64,
                    &queries_f64,
                    max_dist,
                    max_qty,
                ))
            })
        },
    );
}

criterion_group!(benches, v6_codspeed_queries);
criterion_main!(benches);
