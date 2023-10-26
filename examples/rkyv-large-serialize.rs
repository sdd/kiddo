/// Kiddo example: Rkyv large, serialize
///
/// Creates a large tree of random data and serializes it with Rkyv
/// to a file.
///
/// Run the rkyv-large-deserialize example after this to see how
/// quickly memmapped zero-copy deserialization can read in the tree
/// from the file system and perform a query on it.
use elapsed::ElapsedDuration;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use kiddo::{KdTree, SquaredEuclidean};

use kiddo::test_utils::build_populated_tree_float;
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};

const BUFFER_LEN: usize = 10_000_000_000;
const SCRATCH_LEN: usize = 1_000_000_000;

const NUM_ITEMS: usize = 250_000_000;

type Tree = KdTree<f32, 3>;

fn main() -> Result<(), Box<dyn Error>> {
    // create a tree populated with random points
    let start = Instant::now();
    let kdtree: Tree = build_populated_tree_float(NUM_ITEMS, 0);
    println!(
        "Populated kd-tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = [0.123f32, 0.456f32, 0.789f32];
    let nearest_neighbour = kdtree.nearest_one::<SquaredEuclidean>(&query);
    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    let start = Instant::now();

    // create a file for us to serialize into
    let mut file = File::create("./examples/large-random-tree.rkyv")?;

    serialize_to_rkyv(&mut file, kdtree);
    println!(
        "Serialized kd-tree to rkyv file ({})\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}

fn serialize_to_rkyv(file: &mut File, tree: Tree) {
    let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);

    unsafe { serialize_scratch.set_len(SCRATCH_LEN) };
    serialize_buffer.clear();

    let mut serializer = CompositeSerializer::new(
        AlignedSerializer::new(&mut serialize_buffer),
        BufferScratch::new(&mut serialize_scratch),
        Infallible,
    );

    serializer
        .serialize_value(&tree)
        .expect("Could not serialize with rkyv");

    let buf = serializer.into_serializer().into_inner();

    file.write_all(buf)
        .expect("Could not write serialized rkyv to file");
}
