use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::distance::float::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::stem_strategies::DonnellySwPre;
use memmap::MmapOptions;
use rkyv_08::access_unchecked;
use rkyv_08::vec::ArchivedVec;

const QUERY_POINT_QTY: usize = 20_000_000;
const BUCKET_SIZE: usize = 2;

type Tree = ImmutableKdTree<f64, usize, DonnellySwPre<3, 64, 8, 4>, 4, BUCKET_SIZE>;
type ArchivedTree =
    ArchivedR8ImmutableKdTree<f64, usize, DonnellySwPre<3, 64, 8, 4>, 4, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    // faster unsafe ZC Deserialize API
    let start = Instant::now();

    // memmap the tree file into a buffer
    let tree_file = File::open("./examples/immutable-ann-test-tree-rkyv_08.rkyv")?;
    let tree_buf = unsafe { MmapOptions::new().map(&tree_file)? };

    // Get archived tree using unsafe method
    let tree = unsafe { access_unchecked::<ArchivedTree>(&tree_buf) };

    // memmap the tree file into a buffer
    let query_file = File::open("./examples/immutable-ann-test-points-f64-rkyv_08.rkyv")?;
    let query_buf = unsafe { MmapOptions::new().map(&query_file)? };

    // Get archived tree using unsafe method
    let query_points = unsafe { access_unchecked::<ArchivedVec<[f64; 4]>>(&query_buf) };

    println!(
        "Deserialization complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    println!("Performing {QUERY_POINT_QTY:?} random NN queries...");

    let start = Instant::now();
    query_points.iter().for_each(|point| {
        black_box(tree.get_leaf_node_idx(point));
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
