// Legacy filename retained for compatibility. The public KdTree API does not
// expose archived traversal simulation, so this example demonstrates the
// supported public workflow: mmap archived bytes, inspect zero-copy metadata,
// access zero-copy bytes, then run an approximate-NN batch.
use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::kd_tree::ArchivedKdTree;
use kiddo::stem_strategy::EytzingerPf;
use kiddo::SquaredEuclidean;
use memmap::MmapOptions;
use rkyv_08::access;
use rkyv_08::rancor::Error as RkyvError;
use rkyv_08::vec::ArchivedVec;

const BUCKET_SIZE: usize = 2;

type ArchivedTree = ArchivedKdTree<
    f32,
    usize,
    EytzingerPf<4, 4>,
    VecOfArenas<f32, usize, 4, BUCKET_SIZE>,
    4,
    BUCKET_SIZE,
>;

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    let tree_file = File::open("./examples/immutable-test-tree-eytz-f32-rkyv_08.rkyv")?;
    let tree_buf = unsafe { MmapOptions::new().map(&tree_file)? };
    let archived_tree = access::<ArchivedTree, RkyvError>(&tree_buf[..])?;
    println!(
        "Checked zero-copy metadata: size={} leaf_count={} max_stem_level={} max_leaf_len={}",
        archived_tree.size(),
        archived_tree.leaf_count(),
        archived_tree.max_stem_level(),
        archived_tree.max_leaf_len()
    );

    let query_file = File::open("./examples/immutable-test-points-f32-rkyv_08.rkyv")?;
    let query_buf = unsafe { MmapOptions::new().map(&query_file)? };
    let query_points = access::<ArchivedVec<[f32; 4]>, RkyvError>(&query_buf[..])?;

    println!(
        "Zero-copy access complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );
    println!("Running {} approximate NN queries...", query_points.len());

    let start = Instant::now();
    for point in query_points.iter() {
        black_box(archived_tree.approx_nearest_one::<SquaredEuclidean<f32>>(point));
    }
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
