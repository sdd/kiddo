use elapsed::ElapsedDuration;
// use kiddo::immutable::float::kdtree::ImmutableKdTree;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tracing::Level;
use tracing_subscriber::fmt;
use ubyte::ToByteUnit;

use kiddo::float::kdtree::KdTree;
use rkyv::{
    ser::{
        serializers::{AlignedSerializer, BufferScratch, CompositeSerializer},
        Serializer,
    },
    AlignedVec, Infallible,
};

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

fn main() -> Result<(), Box<dyn Error>> {
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // let tree_size = 2usize.pow(23); // ~30s for 8M
    // let seed = 493;

    let tree_size = 2usize.pow(27);
    // let seed = 1;

    //let tree_size = 2usize.pow(3);

    //for seed in 0..1 {
    //00000 {
    //event!(Level::INFO, seed, "NEW SEED");
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let content_to_add: Vec<[f64; 4]> = (0..tree_size).map(|_| rng.gen::<[f64; 4]>()).collect();

    // let start = Instant::now();
    // let tree: ImmutableKdTree<f64, usize, 4, 32> = ImmutableKdTree::optimize_from(&content_to_add);
    // println!(
    //     "Populated ImmutableKdTree with {} items ({})",
    //     tree.size(),
    //     ElapsedDuration::new(start.elapsed())
    // );
    //
    // println!("Tree Stats: {:?}", tree.generate_stats());
    // //}
    //
    // let start = Instant::now();
    // let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    // let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);
    // unsafe {
    //     serialize_scratch.set_len(SCRATCH_LEN);
    // }
    // serialize_buffer.clear();
    // let mut serializer = CompositeSerializer::new(
    //     AlignedSerializer::new(&mut serialize_buffer),
    //     BufferScratch::new(&mut serialize_scratch),
    //     Infallible,
    // );
    // serializer
    //     .serialize_value(&tree)
    //     .expect("Could not serialize with rkyv");
    //
    // let buf = serializer.into_serializer().into_inner();
    // let mut file = File::create("./examples/random-big-immutable.rkyv")?;
    // file.write_all(&buf)
    //     .expect("Could not write serialized rkyv to file");
    //
    // let file_size = file.metadata().unwrap().len().bytes();
    // println!(
    //     "Serialized kd-tree to rkyv file 'random-big-immutable.rkyv' ({}). File size: {:.2}",
    //     ElapsedDuration::new(start.elapsed()),
    //     file_size
    // );

    let start = Instant::now();
    let mut tree2: KdTree<f64, usize, 4, 32, u32> = KdTree::with_capacity(content_to_add.len());
    for (idx, point) in content_to_add.iter().enumerate() {
        tree2.add(point, idx);
    }
    println!(
        "Populated KdTree with {} items ({})",
        content_to_add.len(),
        ElapsedDuration::new(start.elapsed())
    );

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
        .serialize_value(&tree2)
        .expect("Could not serialize with rkyv");
    let buf = serializer.into_serializer().into_inner();
    let mut file2 = File::create("./examples/random-big-mutable.rkyv")?;
    file2
        .write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    let file_size = file2.metadata().unwrap().len().bytes();
    println!(
        "Serialized mutable kd-tree to rkyv file 'random-big-mutable.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    Ok(())
}
