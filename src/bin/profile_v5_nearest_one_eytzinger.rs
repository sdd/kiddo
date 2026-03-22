use kiddo_v5::immutable::float::kdtree::ImmutableKdTree;
use kiddo_v5::SquaredEuclidean;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_POINT_COUNT: usize = 10_000_000;
const DEFAULT_QUERY_COUNT: usize = 1_000;
const DEFAULT_QUERY_BATCH_REPEATS: usize = 2_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;

type Tree = ImmutableKdTree<f64, u32, 3, 32>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let point_count = read_usize_env("KIDDO_PROFILE_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let query_batch_repeats = read_usize_env(
        "KIDDO_PROFILE_QUERY_BATCH_REPEATS",
        DEFAULT_QUERY_BATCH_REPEATS,
    );

    eprintln!(
        "profiling v5 nearest_one: points={} queries={} query_batch_repeats={} point_seed={} query_seed={}",
        point_count, query_count, query_batch_repeats, POINT_SEED, QUERY_SEED
    );

    let mut point_rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    let points: Vec<[f64; 3]> = (0..point_count)
        .map(|_| point_rng.random::<[f64; 3]>())
        .collect();

    let build_start = Instant::now();
    let tree: Tree = ImmutableKdTree::new_from_slice(&points);
    let build_elapsed = build_start.elapsed();

    let mut query_rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    let queries: Vec<[f64; 3]> = (0..query_count)
        .map(|_| query_rng.random::<[f64; 3]>())
        .collect();

    let query_start = Instant::now();
    let mut checksum_dist = 0.0f64;
    let mut checksum_item = 0u64;

    for _ in 0..query_batch_repeats {
        for query in &queries {
            let result = tree.nearest_one::<SquaredEuclidean>(black_box(query));
            checksum_dist += result.distance;
            checksum_item = checksum_item.wrapping_add(result.item as u64);
        }
    }

    let query_elapsed = query_start.elapsed();

    println!(
        "v5 build={:?} query_total={:?} query_per_batch={:?} checksum_dist={:.17e} checksum_item={}",
        build_elapsed,
        query_elapsed,
        query_elapsed / (query_batch_repeats as u32),
        checksum_dist,
        checksum_item
    );
}
