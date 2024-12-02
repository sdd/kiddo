/// Kiddo example: Rkyv large, deserialize
///
/// Deserializes a large tree of random data using memmap and Rkyv's
/// zero-copy deserialization, and then runs a single query against
/// it. Use this to get an idea of the kind of time-to-first-query
/// you can achieve with Kiddo.
///
/// Run the rkyv-large-serialize example before this to generate the
/// tree of random data and serialize it to a file.
use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

use kiddo::{KdTree, SquaredEuclidean};

type Tree = KdTree<f32, 3>;

fn main() -> Result<(), Box<dyn Error>> {
    let query = [0.123f32, 0.456f32, 0.789f32];

    let start = Instant::now();

    // memmap the file into a buffer
    let buf = unsafe { MmapOptions::new().map(&File::open("./examples/large-random-tree.rkyv")?)? };

    // zero-copy deserialize
    let tree = unsafe { rkyv::archived_root::<Tree>(&buf) };
    println!(
        "Memmap ZC Deserialized rkyv file back into a k-d tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // perform a query
    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!("Nearest item to query: {:?}", nearest_neighbour.item);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
