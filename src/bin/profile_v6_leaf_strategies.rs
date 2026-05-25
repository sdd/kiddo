#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
use kiddo::Eytzinger;
use kiddo::SquaredEuclidean;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::{Duration, Instant};

const K: usize = 3;
const B: usize = 32;
const DEFAULT_POINT_COUNT: usize = 262_144;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_QUERY_BATCH_REPEATS: usize = 100;
const POINT_SEED: u64 = 0x5eed_0000_0000_0201;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0202;

type FlatTree = KdTree<f64, u32, Eytzinger<K>, FlatVec<f64, u32, K, B>, K, B>;
type ArenaTree = KdTree<f64, u32, Eytzinger<K>, VecOfArenas<f64, u32, K, B>, K, B>;
type VecOfArraysTree = KdTree<f64, u32, Eytzinger<K>, VecOfArrays<f64, u32, K, B>, K, B>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueryKind {
    Both,
    Nearest,
    Approx,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrategyKind {
    Both,
    Flat,
    Arena,
    VecOfArrays,
}

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_query_kind_env() -> QueryKind {
    match std::env::var("KIDDO_PROFILE_QUERY_KIND")
        .ok()
        .as_deref()
        .unwrap_or("both")
    {
        "nearest" => QueryKind::Nearest,
        "approx" => QueryKind::Approx,
        _ => QueryKind::Both,
    }
}

fn read_strategy_kind_env() -> StrategyKind {
    match std::env::var("KIDDO_PROFILE_STRATEGY")
        .ok()
        .as_deref()
        .unwrap_or("both")
    {
        "flat" => StrategyKind::Flat,
        "arena" => StrategyKind::Arena,
        "voa" | "vec_of_arrays" => StrategyKind::VecOfArrays,
        _ => StrategyKind::Both,
    }
}

fn time_nearest_flat(
    tree: &FlatTree,
    queries: &[[f64; K]],
    repeats: usize,
) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn time_nearest_arena(
    tree: &ArenaTree,
    queries: &[[f64; K]],
    repeats: usize,
) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn time_nearest_vec_of_arrays(
    tree: &VecOfArraysTree,
    queries: &[[f64; K]],
    repeats: usize,
) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn time_approx_flat(tree: &FlatTree, queries: &[[f64; K]], repeats: usize) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn time_approx_arena(
    tree: &ArenaTree,
    queries: &[[f64; K]],
    repeats: usize,
) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn time_approx_vec_of_arrays(
    tree: &VecOfArraysTree,
    queries: &[[f64; K]],
    repeats: usize,
) -> (Duration, f64, u64) {
    let start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..repeats {
        for query in queries {
            let (dist, item) = tree
                .query(black_box(query))
                .nearest_one::<SquaredEuclidean<f64>>()
                .approx()
                .execute();
            checksum_dist += dist;
            checksum_item = checksum_item.wrapping_add(item as u64);
        }
    }

    (start.elapsed(), checksum_dist, checksum_item)
}

fn per_query_ns(elapsed: Duration, queries: usize, repeats: usize) -> f64 {
    elapsed.as_nanos() as f64 / (queries * repeats) as f64
}

fn main() {
    let point_count = read_usize_env("KIDDO_PROFILE_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let query_batch_repeats = read_usize_env(
        "KIDDO_PROFILE_QUERY_BATCH_REPEATS",
        DEFAULT_QUERY_BATCH_REPEATS,
    );
    let query_kind = read_query_kind_env();
    let strategy_kind = read_strategy_kind_env();

    eprintln!(
        "profiling v6 leaf strategies: points={} queries={} repeats={} query_kind={:?} strategy_kind={:?} point_seed={} query_seed={}",
        point_count,
        query_count,
        query_batch_repeats,
        query_kind,
        strategy_kind,
        POINT_SEED,
        QUERY_SEED
    );

    let mut point_rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    let points: Vec<[f64; K]> = (0..point_count)
        .map(|_| point_rng.random::<[f64; K]>())
        .collect();

    let (flat_tree, flat_build_elapsed) =
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Flat) {
            let flat_build_start = Instant::now();
            let flat_tree = KdTree::new_from_slice(&points).unwrap();
            (Some(flat_tree), Some(flat_build_start.elapsed()))
        } else {
            (None, None)
        };

    let (arena_tree, arena_build_elapsed) =
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Arena) {
            let arena_build_start = Instant::now();
            let arena_tree = KdTree::new_from_slice(&points).unwrap();
            (Some(arena_tree), Some(arena_build_start.elapsed()))
        } else {
            (None, None)
        };

    let (vec_of_arrays_tree, vec_of_arrays_build_elapsed) = if matches!(
        strategy_kind,
        StrategyKind::Both | StrategyKind::VecOfArrays
    ) {
        let vec_of_arrays_build_start = Instant::now();
        let vec_of_arrays_tree = KdTree::new_from_slice(&points).unwrap();
        (
            Some(vec_of_arrays_tree),
            Some(vec_of_arrays_build_start.elapsed()),
        )
    } else {
        (None, None)
    };

    let mut query_rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    let queries: Vec<[f64; K]> = (0..query_count)
        .map(|_| query_rng.random::<[f64; K]>())
        .collect();

    if matches!(query_kind, QueryKind::Both | QueryKind::Nearest) {
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Flat) {
            let _ = time_nearest_flat(flat_tree.as_ref().unwrap(), &queries, 1);
        }
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Arena) {
            let _ = time_nearest_arena(arena_tree.as_ref().unwrap(), &queries, 1);
        }
        if matches!(
            strategy_kind,
            StrategyKind::Both | StrategyKind::VecOfArrays
        ) {
            let _ = time_nearest_vec_of_arrays(vec_of_arrays_tree.as_ref().unwrap(), &queries, 1);
        }
    }
    if matches!(query_kind, QueryKind::Both | QueryKind::Approx) {
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Flat) {
            let _ = time_approx_flat(flat_tree.as_ref().unwrap(), &queries, 1);
        }
        if matches!(strategy_kind, StrategyKind::Both | StrategyKind::Arena) {
            let _ = time_approx_arena(arena_tree.as_ref().unwrap(), &queries, 1);
        }
        if matches!(
            strategy_kind,
            StrategyKind::Both | StrategyKind::VecOfArrays
        ) {
            let _ = time_approx_vec_of_arrays(vec_of_arrays_tree.as_ref().unwrap(), &queries, 1);
        }
    }

    println!(
        "build flat={:?} arena={:?} vec_of_arrays={:?}",
        flat_build_elapsed, arena_build_elapsed, vec_of_arrays_build_elapsed
    );

    if matches!(query_kind, QueryKind::Both | QueryKind::Nearest) {
        let nearest_flat = matches!(strategy_kind, StrategyKind::Both | StrategyKind::Flat)
            .then(|| time_nearest_flat(flat_tree.as_ref().unwrap(), &queries, query_batch_repeats));
        let nearest_arena =
            matches!(strategy_kind, StrategyKind::Both | StrategyKind::Arena).then(|| {
                time_nearest_arena(arena_tree.as_ref().unwrap(), &queries, query_batch_repeats)
            });
        let nearest_vec_of_arrays = matches!(
            strategy_kind,
            StrategyKind::Both | StrategyKind::VecOfArrays
        )
        .then(|| {
            time_nearest_vec_of_arrays(
                vec_of_arrays_tree.as_ref().unwrap(),
                &queries,
                query_batch_repeats,
            )
        });

        match (nearest_flat, nearest_arena, nearest_vec_of_arrays) {
            (
                Some((elapsed, dist, item)),
                Some((arena_elapsed, arena_dist, arena_item)),
                Some((vec_of_arrays_elapsed, vec_of_arrays_dist, vec_of_arrays_item)),
            ) => {
                println!(
                    "nearest_one flat={:?} ({:.2} ns/query) arena={:?} ({:.2} ns/query) vec_of_arrays={:?} ({:.2} ns/query) speedups=flat/arena:{:.3}x flat/vec_of_arrays:{:.3}x checksums=({:.17e}, {})/({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    arena_elapsed,
                    per_query_ns(arena_elapsed, query_count, query_batch_repeats),
                    vec_of_arrays_elapsed,
                    per_query_ns(vec_of_arrays_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / arena_elapsed.as_secs_f64(),
                    elapsed.as_secs_f64() / vec_of_arrays_elapsed.as_secs_f64(),
                    dist,
                    item,
                    arena_dist,
                    arena_item,
                    vec_of_arrays_dist,
                    vec_of_arrays_item,
                );
            }
            (Some((elapsed, dist, item)), Some((arena_elapsed, arena_dist, arena_item)), None) => {
                println!(
                    "nearest_one flat={:?} ({:.2} ns/query) arena={:?} ({:.2} ns/query) speedup={:.3}x checksums=({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    arena_elapsed,
                    per_query_ns(arena_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / arena_elapsed.as_secs_f64(),
                    dist,
                    item,
                    arena_dist,
                    arena_item,
                );
            }
            (Some((elapsed, dist, item)), None, Some((voa_elapsed, voa_dist, voa_item))) => {
                println!(
                    "nearest_one flat={:?} ({:.2} ns/query) vec_of_arrays={:?} ({:.2} ns/query) speedup={:.3}x checksums=({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    voa_elapsed,
                    per_query_ns(voa_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / voa_elapsed.as_secs_f64(),
                    dist,
                    item,
                    voa_dist,
                    voa_item,
                );
            }
            (Some((elapsed, dist, item)), None, None) => {
                println!(
                    "nearest_one flat={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, Some((elapsed, dist, item)), None) => {
                println!(
                    "nearest_one arena={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, None, Some((elapsed, dist, item))) => {
                println!(
                    "nearest_one vec_of_arrays={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, None, None) => {}
            _ => {}
        }
    }

    if matches!(query_kind, QueryKind::Both | QueryKind::Approx) {
        let approx_flat = matches!(strategy_kind, StrategyKind::Both | StrategyKind::Flat)
            .then(|| time_approx_flat(flat_tree.as_ref().unwrap(), &queries, query_batch_repeats));
        let approx_arena =
            matches!(strategy_kind, StrategyKind::Both | StrategyKind::Arena).then(|| {
                time_approx_arena(arena_tree.as_ref().unwrap(), &queries, query_batch_repeats)
            });
        let approx_vec_of_arrays = matches!(
            strategy_kind,
            StrategyKind::Both | StrategyKind::VecOfArrays
        )
        .then(|| {
            time_approx_vec_of_arrays(
                vec_of_arrays_tree.as_ref().unwrap(),
                &queries,
                query_batch_repeats,
            )
        });

        match (approx_flat, approx_arena, approx_vec_of_arrays) {
            (
                Some((elapsed, dist, item)),
                Some((arena_elapsed, arena_dist, arena_item)),
                Some((vec_of_arrays_elapsed, vec_of_arrays_dist, vec_of_arrays_item)),
            ) => {
                println!(
                    "approx_nearest_one flat={:?} ({:.2} ns/query) arena={:?} ({:.2} ns/query) vec_of_arrays={:?} ({:.2} ns/query) speedups=flat/arena:{:.3}x flat/vec_of_arrays:{:.3}x checksums=({:.17e}, {})/({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    arena_elapsed,
                    per_query_ns(arena_elapsed, query_count, query_batch_repeats),
                    vec_of_arrays_elapsed,
                    per_query_ns(vec_of_arrays_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / arena_elapsed.as_secs_f64(),
                    elapsed.as_secs_f64() / vec_of_arrays_elapsed.as_secs_f64(),
                    dist,
                    item,
                    arena_dist,
                    arena_item,
                    vec_of_arrays_dist,
                    vec_of_arrays_item,
                );
            }
            (Some((elapsed, dist, item)), Some((arena_elapsed, arena_dist, arena_item)), None) => {
                println!(
                    "approx_nearest_one flat={:?} ({:.2} ns/query) arena={:?} ({:.2} ns/query) speedup={:.3}x checksums=({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    arena_elapsed,
                    per_query_ns(arena_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / arena_elapsed.as_secs_f64(),
                    dist,
                    item,
                    arena_dist,
                    arena_item,
                );
            }
            (Some((elapsed, dist, item)), None, Some((voa_elapsed, voa_dist, voa_item))) => {
                println!(
                    "approx_nearest_one flat={:?} ({:.2} ns/query) vec_of_arrays={:?} ({:.2} ns/query) speedup={:.3}x checksums=({:.17e}, {})/({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    voa_elapsed,
                    per_query_ns(voa_elapsed, query_count, query_batch_repeats),
                    elapsed.as_secs_f64() / voa_elapsed.as_secs_f64(),
                    dist,
                    item,
                    voa_dist,
                    voa_item,
                );
            }
            (Some((elapsed, dist, item)), None, None) => {
                println!(
                    "approx_nearest_one flat={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, Some((elapsed, dist, item)), None) => {
                println!(
                    "approx_nearest_one arena={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, None, Some((elapsed, dist, item))) => {
                println!(
                    "approx_nearest_one vec_of_arrays={:?} ({:.2} ns/query) checksum=({:.17e}, {})",
                    elapsed,
                    per_query_ns(elapsed, query_count, query_batch_repeats),
                    dist,
                    item,
                );
            }
            (None, None, None) => {}
            _ => {}
        }
    }
}
