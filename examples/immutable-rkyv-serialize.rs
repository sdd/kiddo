use elapsed::ElapsedDuration;
// use memmap::MmapOptions;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;
use ubyte::ToByteUnit;

use kiddo::float::distance::SquaredEuclidean;
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};

use kiddo::immutable::float::kdtree::{ImmutableKdTree, ImmutableKdTreeRK};

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

const NUM_ITEMS: usize = 50_000_000;

type Tree = ImmutableKdTree<f64, u32, 3, 256>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    let query = [0.123f64, 0.456f64, 0.789f64];

    // build and serialize a large ImmutableKdTree
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let content_to_add: Vec<[f64; 3]> = (0..NUM_ITEMS).map(|_| rng.random::<[f64; 3]>()).collect();

    let start = Instant::now();
    let tree: Tree = ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Populated ImmutableKdTree with {} items ({})",
        tree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    let start = Instant::now();

    let tree_rk: ImmutableKdTreeRK<f64, u32, 3, 256> = tree.into();

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
        .serialize_value(&tree_rk)
        .expect("Could not serialize with rkyv");

    let buf = serializer.into_serializer().into_inner();
    let mut file = File::create("./examples/immutable-test-tree.rkyv")?;
    file.write_all(buf)
        .expect("Could not write serialized rkyv to file");

    let file_size = file.metadata().unwrap().len().bytes();
    println!(
        "Serialized k-d tree to rkyv file 'immutable-test-tree.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    Ok(())
}
