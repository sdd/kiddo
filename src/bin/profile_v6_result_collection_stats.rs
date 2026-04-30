use kiddo::dist::SquaredEuclidean;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::results::result_collection_stats::{reset, snapshot, ResultCollectionStats};
use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
use kiddo::stem_strategy::Eytzinger;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_REPEATS: usize = 1;
const DEFAULT_MAX_QTY: usize = 16;
const DEFAULT_MAX_DIST: f64 = 0.0025;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type ArchivedEytzingerTree =
    kiddo::kd_tree::ArchivedKdTree<f64, u32, Eytzinger<K>, ArenaLeaves, K, B>;
type ArchivedDonnellyPfTree =
    kiddo::kd_tree::ArchivedKdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;

#[derive(Clone, Copy)]
struct RunResult {
    elapsed_ns: f64,
    checksum_len: usize,
    checksum_item: u64,
    checksum_dist: f64,
    stats: ResultCollectionStats,
}

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

fn print_stats(label: &str, result: RunResult, total_queries: usize) {
    let total_queries_f = total_queries as f64;
    let stats = result.stats;
    let avg_flush_size = if stats.buffer_flushes == 0 {
        0.0
    } else {
        stats.buffer_flush_size_sum as f64 / stats.buffer_flushes as f64
    };
    let avg_insert_position = if stats.sorted_insert_calls == 0 {
        0.0
    } else {
        stats.sorted_insert_position_sum as f64 / stats.sorted_insert_calls as f64
    };
    let avg_shifted_items = if stats.sorted_insert_calls == 0 {
        0.0
    } else {
        stats.sorted_shifted_items_sum as f64 / stats.sorted_insert_calls as f64
    };

    println!(
        "{}: {:.2} ns/query checksums=(len={}, item={}, dist={:.17e})",
        label,
        result.elapsed_ns / total_queries_f,
        result.checksum_len,
        result.checksum_item,
        result.checksum_dist
    );
    println!(
        "  leaf_visits={} ({:.4}/query)",
        stats.leaf_visits,
        stats.leaf_visits as f64 / total_queries_f
    );
    println!(
        "  leaf_visits_before_full={} after_full={} full_transitions={}",
        stats.leaf_visits_before_full,
        stats.leaf_visits_after_full,
        stats.collection_full_transitions
    );
    println!(
        "  candidates_emitted={} ({:.4}/query)",
        stats.candidates_emitted,
        stats.candidates_emitted as f64 / total_queries_f
    );
    println!(
        "  candidates_emitted_before_full={} after_full={}",
        stats.candidates_emitted_before_full, stats.candidates_emitted_after_full
    );
    println!(
        "  best_item_threshold_rejects={}",
        stats.best_item_threshold_rejects
    );
    println!(
        "  buffer_flushes={} ({:.4}/query) avg_size={:.4} max_size={}",
        stats.buffer_flushes,
        stats.buffer_flushes as f64 / total_queries_f,
        avg_flush_size,
        stats.buffer_flush_size_max
    );
    println!(
        "  flush_buckets: 0={} 1={} 2_4={} 5_8={} 9_plus={}",
        stats.buffer_flush_size_0,
        stats.buffer_flush_size_1,
        stats.buffer_flush_size_2_4,
        stats.buffer_flush_size_5_8,
        stats.buffer_flush_size_9_plus
    );
    println!(
        "  collector_add_calls={} ({:.4}/query)",
        stats.collector_add_calls,
        stats.collector_add_calls as f64 / total_queries_f
    );
    println!(
        "  collector_add_all_calls={} ({:.4}/query) entries={}",
        stats.collector_add_all_calls,
        stats.collector_add_all_calls as f64 / total_queries_f,
        stats.collector_add_all_entry_count
    );
    println!(
        "  threshold_distance_calls={} full={} some={}",
        stats.threshold_distance_calls,
        stats.threshold_distance_full,
        stats.threshold_distance_some
    );
    println!(
        "  sorted_insert_calls={} avg_pos={:.4} avg_shifted={:.4}",
        stats.sorted_insert_calls, avg_insert_position, avg_shifted_items
    );
    println!(
        "  heap_adds: pushes={} replacements={} discards={}",
        stats.heap_add_pushes, stats.heap_add_replacements, stats.heap_add_discards
    );
    println!(
        "  query_stack: pushes={} pops={} prunes={}",
        stats.query_stack_pushes, stats.query_stack_pops, stats.query_prunes
    );
}

fn print_strategy_results(label: &str, sorted: RunResult, best: RunResult, total_queries: usize) {
    println!();
    println!("strategy={label}");
    print_stats("sorted nearest_n_within", sorted, total_queries);
    print_stats("best_n_within", best, total_queries);
}

fn archive_path(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}-{suffix}.rkyv", prefix.display()))
}

fn load_aligned_archive(
    path: &Path,
) -> Result<rkyv_08::util::AlignedVec<128>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let mut aligned = rkyv_08::util::AlignedVec::<128>::with_capacity(bytes.len());
    aligned.extend_from_slice(&bytes);
    Ok(aligned)
}

fn run_sorted_nearest_n_within_archived_donnelly(
    tree: &ArchivedDonnellyPfTree,
    queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
    repeats: usize,
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for _ in 0..repeats {
        for query in queries.iter() {
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
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_len,
        checksum_item,
        checksum_dist,
        stats: snapshot(),
    }
}

fn run_best_n_within_archived_donnelly(
    tree: &ArchivedDonnellyPfTree,
    queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
    repeats: usize,
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for _ in 0..repeats {
        for query in queries.iter() {
            let results =
                tree.best_n_within::<SquaredEuclidean<f64>>(black_box(query), max_dist, max_qty);
            checksum_len += results.len();

            for result in results.into_vec() {
                checksum_item = checksum_item.wrapping_add(result.item as u64);
                checksum_dist += result.distance;
            }
        }
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_len,
        checksum_item,
        checksum_dist,
        stats: snapshot(),
    }
}

fn run_sorted_nearest_n_within_archived_eytzinger(
    tree: &ArchivedEytzingerTree,
    queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
    repeats: usize,
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for _ in 0..repeats {
        for query in queries.iter() {
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
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_len,
        checksum_item,
        checksum_dist,
        stats: snapshot(),
    }
}

fn run_best_n_within_archived_eytzinger(
    tree: &ArchivedEytzingerTree,
    queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
    repeats: usize,
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> RunResult {
    reset();
    let start = Instant::now();
    let mut checksum_len = 0usize;
    let mut checksum_item = 0u64;
    let mut checksum_dist = 0.0f64;

    for _ in 0..repeats {
        for query in queries.iter() {
            let results =
                tree.best_n_within::<SquaredEuclidean<f64>>(black_box(query), max_dist, max_qty);
            checksum_len += results.len();

            for result in results.into_vec() {
                checksum_item = checksum_item.wrapping_add(result.item as u64);
                checksum_dist += result.distance;
            }
        }
    }

    RunResult {
        elapsed_ns: start.elapsed().as_nanos() as f64,
        checksum_len,
        checksum_item,
        checksum_dist,
        stats: snapshot(),
    }
}

fn run_archived_profile(
    repeats: usize,
    max_dist: f64,
    max_qty: NonZeroUsize,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = PathBuf::from(
        std::env::var("KIDDO_PROFILE_ARCHIVE_PREFIX")
            .unwrap_or_else(|_| "./target/kiddo-profile-v6-result-collection".to_owned()),
    );
    let eytzinger_path = archive_path(&prefix, "eytzinger-tree");
    let donnelly_path = archive_path(&prefix, "donnelly-pf-tree");
    let queries_path = archive_path(&prefix, "queries");

    eprintln!(
        "loading profile archives: eytzinger={} donnelly_pf={} queries={}",
        eytzinger_path.display(),
        donnelly_path.display(),
        queries_path.display()
    );

    let load_start = Instant::now();
    let eytzinger_bytes = load_aligned_archive(&eytzinger_path)?;
    let donnelly_bytes = load_aligned_archive(&donnelly_path)?;
    let query_bytes = load_aligned_archive(&queries_path)?;
    let eytzinger_tree =
        rkyv_08::access::<ArchivedEytzingerTree, rkyv_08::rancor::Error>(&eytzinger_bytes[..])?;
    let donnelly_pf_tree =
        rkyv_08::access::<ArchivedDonnellyPfTree, rkyv_08::rancor::Error>(&donnelly_bytes[..])?;
    let queries = rkyv_08::access::<rkyv_08::vec::ArchivedVec<[f64; K]>, rkyv_08::rancor::Error>(
        &query_bytes[..],
    )?;
    eprintln!(
        "archive_load_ns={:.0} queries={} eytzinger_size={} donnelly_size={}",
        load_start.elapsed().as_nanos() as f64,
        queries.len(),
        eytzinger_tree.size(),
        donnelly_pf_tree.size()
    );

    let total_queries = queries.len() * repeats;
    let eytzinger_sorted = run_sorted_nearest_n_within_archived_eytzinger(
        eytzinger_tree,
        queries,
        repeats,
        max_dist,
        max_qty,
    );
    let eytzinger_best =
        run_best_n_within_archived_eytzinger(eytzinger_tree, queries, repeats, max_dist, max_qty);
    print_strategy_results(
        "Eytzinger archived",
        eytzinger_sorted,
        eytzinger_best,
        total_queries,
    );

    let donnelly_pf_sorted = run_sorted_nearest_n_within_archived_donnelly(
        donnelly_pf_tree,
        queries,
        repeats,
        max_dist,
        max_qty,
    );
    let donnelly_pf_best =
        run_best_n_within_archived_donnelly(donnelly_pf_tree, queries, repeats, max_dist, max_qty);
    print_strategy_results(
        "Donnelly PF archived",
        donnelly_pf_sorted,
        donnelly_pf_best,
        total_queries,
    );

    Ok(())
}

fn main() {
    let repeats = read_usize_env("KIDDO_PROFILE_QUERY_BATCH_REPEATS", DEFAULT_REPEATS);
    let max_qty =
        NonZeroUsize::new(read_usize_env("KIDDO_PROFILE_MAX_QTY", DEFAULT_MAX_QTY)).unwrap();
    let max_dist = read_f64_env("KIDDO_PROFILE_MAX_DIST", DEFAULT_MAX_DIST);

    if let Err(err) = run_archived_profile(repeats, max_dist, max_qty) {
        eprintln!("archived profile failed: {err}");
        std::process::exit(1);
    }
}
