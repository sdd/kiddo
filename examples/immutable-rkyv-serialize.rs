use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tracing::Level;
use tracing_subscriber::fmt;
use ubyte::ToByteUnit;

use kiddo::float::distance::SquaredEuclidean;
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};

use kiddo::immutable::float::kdtree::ImmutableKdTree;

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

const NUM_ITEMS: usize = 2_500;
const BUCKET_SIZE: usize = 32;

type Tree = ImmutableKdTree<f64, u32, 3, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // build and serialize small tree for ArchivedImmutableKdTree doctests
    let content: Vec<[f64; 3]> = vec![[1.0, 2.0, 5.0], [2.0, 3.0, 6.0]];
    let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);
    let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);
    unsafe {
        serialize_scratch.set_len(SCRATCH_LEN);
    }
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
    let mut file = File::create("./examples/immutable-doctest-tree.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    // build and serialize a larger tree
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let content_to_add: Vec<[f64; 3]> = (0..NUM_ITEMS).map(|_| rng.gen::<[f64; 3]>()).collect();

    let start = Instant::now();
    let tree: Tree = ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Populated ImmutableKdTree with {} items ({})",
        tree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    println!("Tree Stats: {:?}", tree.generate_stats());

    let start = Instant::now();
    let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);
    unsafe {
        serialize_scratch.set_len(SCRATCH_LEN);
    }
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
    let mut file = File::create("./examples/immutable-test-tree.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    let file_size = file.metadata().unwrap().len().bytes();
    println!(
        "Serialized kd-tree to rkyv file 'immutable-test-tree.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    let start = Instant::now();

    // memmap the file into a buffer
    let buf =
        unsafe { MmapOptions::new().map(&File::open("./examples/immutable-test-tree.rkyv")?)? };

    // zero-copy deserialize
    let tree = unsafe { rkyv::archived_root::<Tree>(&buf) };
    println!(
        "Memmap ZC Deserialized rkyv file back into a kd-tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // perform a query
    let query = [0.123f64, 0.456f64, 0.789f64];
    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!("Nearest item to query: {:?}", nearest_neighbour.item);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
