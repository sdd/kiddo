#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::{Block3, Donnelly, DonnellySimdFull};
use kiddo::test_utils::exact_query_stats::{reset, snapshot, ExactQueryStats};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_POINT_COUNT: usize = 1usize << 22;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const DEFAULT_REPEATS: usize = 1;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type DonnellyTree = KdTree<f64, u32, Donnelly<Block3>, ArenaLeaves, K, B>;
type DonnellySimdTree = KdTree<f64, u32, DonnellySimdFull<Block3>, ArenaLeaves, K, B>;

#[derive(Clone, Copy)]
struct RunResult {
    elapsed_ns: f64,
    checksum_dist: f64,
    checksum_item: u64,
    stats: ExactQueryStats,
}

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

fn run_donnelly(tree: &DonnellyTree, queries: &[[f64; K]], repeats: usize) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let result = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_dist,
        checksum_item,
        stats: snapshot(),
    }
}

fn run_donnelly_simd(tree: &DonnellySimdTree, queries: &[[f64; K]], repeats: usize) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let result = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_dist,
        checksum_item,
        stats: snapshot(),
    }
}

fn print_stats(label: &str, result: RunResult, total_queries: usize) {
    let total_queries_f = total_queries as f64;
    let stats = result.stats;

    println!(
        "{}: {:.2} ns/query checksums=({:.17e}, {})",
        label,
        result.elapsed_ns / total_queries_f,
        result.checksum_dist,
        result.checksum_item
    );
    println!(
        "  leaf_visits={} ({:.4}/query)",
        stats.leaf_visits,
        stats.leaf_visits as f64 / total_queries_f
    );
    println!(
        "  scalar_stack_pops={} ({:.4}/query)",
        stats.scalar_stack_pops,
        stats.scalar_stack_pops as f64 / total_queries_f
    );
    println!(
        "  simd_single_pops={} ({:.4}/query)",
        stats.simd_single_pops,
        stats.simd_single_pops as f64 / total_queries_f
    );
    println!("  simd_stack_max_len={}", stats.simd_stack_max_len);
    println!(
        "  block3_pending_pops={} ({:.4}/query)",
        stats.block3_pending_pops,
        stats.block3_pending_pops as f64 / total_queries_f
    );
    println!(
        "  block3_pending_mask_bits={} ({:.4}/pending-pop)",
        stats.block3_pending_mask_bits,
        if stats.block3_pending_pops == 0 {
            0.0
        } else {
            stats.block3_pending_mask_bits as f64 / stats.block3_pending_pops as f64
        }
    );
    println!(
        "  block3_candidate_mask_bits={} ({:.4}/pending-pop, nonzero={})",
        stats.block3_candidate_mask_bits,
        if stats.block3_pending_pops == 0 {
            0.0
        } else {
            stats.block3_candidate_mask_bits as f64 / stats.block3_pending_pops as f64
        },
        stats.block3_candidate_mask_nonzero
    );
    println!(
        "  block3_step_entries={} ({:.4}/query)",
        stats.block3_step_entries,
        stats.block3_step_entries as f64 / total_queries_f
    );
    println!(
        "  block3_full_steps={} ({:.4}/query)",
        stats.block3_full_steps,
        stats.block3_full_steps as f64 / total_queries_f
    );
    println!(
        "  block3_scalar_fallback_steps={} ({:.4}/query)",
        stats.block3_scalar_fallback_steps,
        stats.block3_scalar_fallback_steps as f64 / total_queries_f
    );
}

fn main() {
    let point_count = read_usize_env("KIDDO_PROFILE_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let repeats = read_usize_env("KIDDO_PROFILE_QUERY_BATCH_REPEATS", DEFAULT_REPEATS);

    eprintln!(
        "profiling v6 stem exact stats: points={} queries={} repeats={} point_seed={} query_seed={}",
        point_count,
        query_count,
        repeats,
        POINT_SEED,
        QUERY_SEED
    );

    let points = build_points(point_count);
    let queries = build_queries(query_count);

    let build_start = Instant::now();
    let donnelly_tree: DonnellyTree = KdTree::new_from_slice(&points).unwrap();
    let donnelly_build_ns = build_start.elapsed().as_nanos() as f64;

    let build_start = Instant::now();
    let donnelly_simd_tree: DonnellySimdTree = KdTree::new_from_slice(&points).unwrap();
    let donnelly_simd_build_ns = build_start.elapsed().as_nanos() as f64;

    eprintln!(
        "build ns/query-batch: donnelly={:.0} blocksimd={:.0}",
        donnelly_build_ns, donnelly_simd_build_ns
    );

    let total_queries = query_count * repeats;
    let donnelly = run_donnelly(&donnelly_tree, &queries, repeats);
    let donnelly_simd = run_donnelly_simd(&donnelly_simd_tree, &queries, repeats);

    print_stats("Donnelly", donnelly, total_queries);
    print_stats("DonnellySimdFull", donnelly_simd, total_queries);
}
