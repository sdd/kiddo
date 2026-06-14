#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use kiddo::dist::{DistanceMetricScalar, SquaredEuclidean};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::{Block3, Donnelly, DonnellyMarkerScalar, DonnellyMarkerSimd};
use kiddo::test_utils::exact_query_stats::{reset, snapshot, ExactQueryStats};
use kiddo::test_utils::exact_query_trace;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_POINT_COUNT: usize = 1usize << 22;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type DonnellyTree = KdTree<f64, u32, Donnelly<3, 64, 8, K>, ArenaLeaves, K, B>;
type DonnellyBlockScalarTree =
    KdTree<f64, u32, DonnellyMarkerScalar<Block3, 64, 8, K>, ArenaLeaves, K, B>;
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

fn squared_euclidean_dist(a: &[f64; K], b: &[f64; K]) -> f64 {
    let aw = (*a).map(<SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::widen_coord);
    let bw = (*b).map(<SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::widen_coord);
    <SquaredEuclidean<f64> as DistanceMetricScalar<f64>>::dist::<K>(&aw, &bw)
}

fn linear_search(points: &[[f64; K]], query: &[f64; K]) -> (f64, u32) {
    let mut best_dist = f64::INFINITY;
    let mut best_item = u32::MAX;

    for (idx, point) in points.iter().enumerate() {
        let dist = squared_euclidean_dist(point, query);
        if dist < best_dist {
            best_dist = dist;
            best_item = idx as u32;
        }
    }

    (best_dist, best_item)
}

fn run_donnelly(tree: &DonnellyTree, query: &[f64; K]) -> ((f64, u32), ExactQueryStats) {
    reset();
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    ((result.distance, result.item), snapshot())
}

fn run_donnelly_block_scalar(
    tree: &DonnellyBlockScalarTree,
    query: &[f64; K],
) -> ((f64, u32), ExactQueryStats) {
    reset();
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    ((result.distance, result.item), snapshot())
}

fn run_donnelly_simd(
    tree: &DonnellySimdTree,
    query: &[f64; K],
    force_block_step: bool,
) -> ((f64, u32), ExactQueryStats) {
    reset();
    if force_block_step {
        unsafe { std::env::set_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP", "1") };
    } else {
        unsafe { std::env::remove_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP") };
    }
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    unsafe { std::env::remove_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP") };
    ((result.distance, result.item), snapshot())
}

fn run_donnelly_trace(
    tree: &DonnellyTree,
    query: &[f64; K],
) -> (
    (f64, u32),
    ExactQueryStats,
    Vec<kiddo::test_utils::exact_query_trace::ExactQueryTraceEvent>,
) {
    reset();
    exact_query_trace::set_enabled(true);
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    let stats = snapshot();
    let trace = exact_query_trace::snapshot();
    exact_query_trace::set_enabled(false);
    ((result.distance, result.item), stats, trace)
}

fn run_donnelly_block_scalar_trace(
    tree: &DonnellyBlockScalarTree,
    query: &[f64; K],
) -> (
    (f64, u32),
    ExactQueryStats,
    Vec<kiddo::test_utils::exact_query_trace::ExactQueryTraceEvent>,
) {
    reset();
    exact_query_trace::set_enabled(true);
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    let stats = snapshot();
    let trace = exact_query_trace::snapshot();
    exact_query_trace::set_enabled(false);
    ((result.distance, result.item), stats, trace)
}

fn run_donnelly_simd_trace(
    tree: &DonnellySimdTree,
    query: &[f64; K],
    force_block_step: bool,
) -> (
    (f64, u32),
    ExactQueryStats,
    Vec<kiddo::test_utils::exact_query_trace::ExactQueryTraceEvent>,
) {
    reset();
    exact_query_trace::set_enabled(true);
    if force_block_step {
        unsafe { std::env::set_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP", "1") };
    } else {
        unsafe { std::env::remove_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP") };
    }
    let result = tree
        .query(query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    unsafe { std::env::remove_var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP") };
    let stats = snapshot();
    let trace = exact_query_trace::snapshot();
    exact_query_trace::set_enabled(false);
    ((result.distance, result.item), stats, trace)
}

fn same_result(lhs: (f64, u32), rhs: (f64, u32)) -> bool {
    lhs.1 == rhs.1 && lhs.0.to_bits() == rhs.0.to_bits()
}

fn main() {
    let point_count = read_usize_env("KIDDO_REPRO_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_REPRO_QUERIES", DEFAULT_QUERY_COUNT);

    eprintln!(
        "repro donnelly block3 exact divergence: points={} queries={} point_seed={} query_seed={}",
        point_count, query_count, POINT_SEED, QUERY_SEED
    );

    let points = build_points(point_count);
    let queries = build_queries(query_count);

    let donnelly_tree: DonnellyTree = KdTree::new_from_slice(&points).unwrap();
    let donnelly_block_scalar_tree: DonnellyBlockScalarTree =
        KdTree::new_from_slice(&points).unwrap();
    let donnelly_simd_tree: DonnellySimdTree = KdTree::new_from_slice(&points).unwrap();

    for (query_idx, query) in queries.iter().enumerate() {
        let expected = linear_search(&points, query);
        let (donnelly, donnelly_stats) = run_donnelly(&donnelly_tree, query);
        let (donnelly_block_scalar, donnelly_block_scalar_stats) =
            run_donnelly_block_scalar(&donnelly_block_scalar_tree, query);
        let (simd_current, simd_current_stats) =
            run_donnelly_simd(&donnelly_simd_tree, query, false);
        let (simd_forced, simd_forced_stats) = run_donnelly_simd(&donnelly_simd_tree, query, true);

        if !same_result(donnelly_block_scalar, simd_forced) {
            let (donnelly_trace_result, donnelly_trace_stats, donnelly_trace) =
                run_donnelly_trace(&donnelly_tree, query);
            let (
                donnelly_block_scalar_trace_result,
                donnelly_block_scalar_trace_stats,
                donnelly_block_scalar_trace,
            ) = run_donnelly_block_scalar_trace(&donnelly_block_scalar_tree, query);
            let (simd_current_trace_result, simd_current_trace_stats, simd_current_trace) =
                run_donnelly_simd_trace(&donnelly_simd_tree, query, false);
            let (simd_forced_trace_result, simd_forced_trace_stats, simd_forced_trace) =
                run_donnelly_simd_trace(&donnelly_simd_tree, query, true);
            println!("first divergence at query_idx={}", query_idx);
            println!("query={:?}", query);
            println!("linear_search={:?}", expected);
            println!("donnelly={:?}", donnelly);
            println!("donnelly_stats={:?}", donnelly_stats);
            println!("donnelly_block_scalar={:?}", donnelly_block_scalar);
            println!(
                "donnelly_block_scalar_stats={:?}",
                donnelly_block_scalar_stats
            );
            println!("simd_current={:?}", simd_current);
            println!("simd_current_stats={:?}", simd_current_stats);
            println!("simd_forced={:?}", simd_forced);
            println!("simd_forced_stats={:?}", simd_forced_stats);
            println!("donnelly_trace_result={:?}", donnelly_trace_result);
            println!("donnelly_trace_stats={:?}", donnelly_trace_stats);
            println!("donnelly_trace:");
            for (idx, event) in donnelly_trace.iter().enumerate() {
                println!("  [{}] {:?}", idx, event);
            }
            println!(
                "donnelly_block_scalar_trace_result={:?}",
                donnelly_block_scalar_trace_result
            );
            println!(
                "donnelly_block_scalar_trace_stats={:?}",
                donnelly_block_scalar_trace_stats
            );
            println!("donnelly_block_scalar_trace:");
            for (idx, event) in donnelly_block_scalar_trace.iter().enumerate() {
                println!("  [{}] {:?}", idx, event);
            }
            println!("simd_current_trace_result={:?}", simd_current_trace_result);
            println!("simd_current_trace_stats={:?}", simd_current_trace_stats);
            println!("simd_current_trace:");
            for (idx, event) in simd_current_trace.iter().enumerate() {
                println!("  [{}] {:?}", idx, event);
            }
            println!("simd_forced_trace_result={:?}", simd_forced_trace_result);
            println!("simd_forced_trace_stats={:?}", simd_forced_trace_stats);
            println!("simd_forced_trace:");
            for (idx, event) in simd_forced_trace.iter().enumerate() {
                println!("  [{}] {:?}", idx, event);
            }
            return;
        }
    }

    println!(
        "no divergence found in first {} queries (forced block-step path)",
        query_count
    );
}
