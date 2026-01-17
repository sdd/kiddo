use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use elapsed::ElapsedDuration;
use rand::{Rng, SeedableRng};
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
use ubyte::ToByteUnit;

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::stem_strategies::Eytzinger;

const TREE_SIZE: usize = 2usize.pow(25);
const BUCKET_SIZE: usize = 2;

type Tree = ImmutableKdTree<f32, usize, Eytzinger<4>, 4, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

    let start = Instant::now();
    println!("Building an optimized tree of {TREE_SIZE:?} items...");
    let tree: Tree = ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();

    let buf = to_bytes::<RkyvError>(&tree)?;

    let mut file = File::create("./examples/immutable-test-tree-eytz-f32-rkyv_08.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    let file_size = file.metadata().unwrap().len().bytes();
    println!(
        "Serialized k-d tree to rkyv file 'immutable-test-tree-eytz-f32-rkyv_08.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    println!(
        "Serialization complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // println!("Generating {QUERY_POINT_QTY:?} random NN queries...");
    // let query_points: Vec<[f32; 4]> = build_query_points_float(QUERY_POINT_QTY);
    //
    // let buf = to_bytes::<RkyvError>(&query_points)?;
    //
    // let mut file = File::create("./examples/immutable-ann-test-points-eytz-rkyv_08.rkyv")?;
    // file.write_all(&buf)
    //     .expect("Could not write serialized rkyv to file");
    //

    Ok(())
}
