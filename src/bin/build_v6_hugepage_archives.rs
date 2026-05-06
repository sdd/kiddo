#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rkyv_08::api::high::to_bytes_in;
use rkyv_08::rancor::Error as RkyvError;
use rkyv_08::util::AlignedVec;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_POINT_COUNT: usize = 1usize << 25;
const DEFAULT_QUERY_COUNT: usize = 100_000;
const POINT_SEED: u64 = 0x5eed_0000_0000_0401;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0402;

type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
type DonnellyPfTree = KdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn archive_path(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}-{suffix}.rkyv", prefix.display()))
}

fn build_points(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn write_archive_bytes(
    label: &str,
    bytes: &AlignedVec<128>,
    path: &Path,
    elapsed_ns: f64,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &bytes[..])?;
    eprintln!(
        "wrote {label}: {} bytes to {} ({elapsed_ns:.0} ns)",
        bytes.len(),
        path.display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let point_count = read_usize_env("KIDDO_PROFILE_POINTS", DEFAULT_POINT_COUNT);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let prefix = PathBuf::from(
        std::env::var("KIDDO_PROFILE_ARCHIVE_PREFIX")
            .unwrap_or_else(|_| "./target/kiddo-hugepage-v6".to_owned()),
    );

    eprintln!(
        "building v6 huge-page profile archives: points={} queries={} prefix={} point_seed={} query_seed={}",
        point_count,
        query_count,
        prefix.display(),
        POINT_SEED,
        QUERY_SEED
    );

    let start = Instant::now();
    let points = build_points(point_count);
    eprintln!(
        "generated points in {:.0} ns",
        start.elapsed().as_nanos() as f64
    );

    let start = Instant::now();
    let tree: DonnellyPfTree = KdTree::new_from_slice(&points);
    eprintln!(
        "built Donnelly PF tree in {:.0} ns",
        start.elapsed().as_nanos() as f64
    );
    drop(points);

    let start = Instant::now();
    let bytes = to_bytes_in::<_, RkyvError>(&tree, AlignedVec::<128>::new())?;
    write_archive_bytes(
        "Donnelly PF tree",
        &bytes,
        &archive_path(&prefix, "donnelly-pf-tree"),
        start.elapsed().as_nanos() as f64,
    )?;
    drop(bytes);
    drop(tree);

    let start = Instant::now();
    let queries = build_queries(query_count);
    eprintln!(
        "generated queries in {:.0} ns",
        start.elapsed().as_nanos() as f64
    );

    let start = Instant::now();
    let bytes = to_bytes_in::<_, RkyvError>(&queries, AlignedVec::<128>::new())?;
    write_archive_bytes(
        "queries",
        &bytes,
        &archive_path(&prefix, "queries"),
        start.elapsed().as_nanos() as f64,
    )?;

    Ok(())
}
