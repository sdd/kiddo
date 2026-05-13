// Legacy filename retained for compatibility. This version uses the current
// rkyv tree combination: KdTree + VecOfArenas + EytzingerPf.
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::EytzingerPf;
use rand::{Rng, SeedableRng};
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
use ubyte::ToByteUnit;

const TREE_SIZE: usize = 2usize.pow(25);
const QUERY_POINT_QTY: usize = 20_000_000;
const BUCKET_SIZE: usize = 2;

type Tree =
    KdTree<f64, usize, EytzingerPf<4, 8>, VecOfArenas<f64, usize, 4, BUCKET_SIZE>, 4, BUCKET_SIZE>;

fn build_query_points(count: usize) -> Vec<[f64; 4]> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    (0..count).map(|_| rng.random::<[f64; 4]>()).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let points: Vec<[f64; 4]> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

    let start = Instant::now();
    println!("Building a tree of {TREE_SIZE} items...");
    let tree: Tree = KdTree::new_from_slice(&points)?;
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    let buf = to_bytes::<RkyvError>(&tree)?;
    let mut file = File::create("./examples/immutable-ann-test-tree-dy-f64-rkyv_08.rkyv")?;
    file.write_all(&buf)?;
    let file_size = file.metadata()?.len().bytes();
    println!(
        "Serialized tree to 'immutable-ann-test-tree-dy-f64-rkyv_08.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    println!("Generating {QUERY_POINT_QTY} random query points...");
    let query_points = build_query_points(QUERY_POINT_QTY);
    let buf = to_bytes::<RkyvError>(&query_points)?;
    let mut file = File::create("./examples/immutable-ann-test-points-f64-rkyv_08.rkyv")?;
    file.write_all(&buf)?;

    Ok(())
}
