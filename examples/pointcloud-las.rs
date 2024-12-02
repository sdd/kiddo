/// Kiddo example: Pointcloud LAS file
///
/// Populates a Kiddo KdTree from a pointcloud LAS file
/// An example LAZ (compressed LAS) file can be found here:
/// https://cesium.com/public/learn/House.laz
use elapsed::ElapsedDuration;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use kiddo::SquaredEuclidean;
use las::Reader;

use kiddo::immutable::float::kdtree::{ImmutableKdTree, ImmutableKdTreeRK};
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};
use tracing::Level;
use tracing_subscriber::fmt;

const BUFFER_LEN: usize = 10_000_000_000;
const SCRATCH_LEN: usize = 1_000_000_000;

type Tree = ImmutableKdTree<f32, u32, 3, 64>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::WARN).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    // create a tree populated with random points
    let start = Instant::now();

    let mut reader = Reader::from_path("./House.laz")?;

    let points: Vec<[f32; 3]> = reader
        .points()
        .map(|point| {
            let point = point.unwrap();
            [point.x as f32, point.y as f32, point.z as f32]
        })
        .collect();

    println!("Points loaded from LAZ file. Count: {}", points.len());

    let kdtree: Tree = (&*points).into();

    println!(
        "Populated k-d tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = [0.123f32, 0.456f32, 0.789f32];
    let nearest_neighbour = kdtree.nearest_one::<SquaredEuclidean>(&query);
    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    let start = Instant::now();

    // create a file for us to serialize into
    let mut file = File::create("./examples/house.rkyv")?;

    serialize_to_rkyv(&mut file, kdtree);
    println!(
        "Serialized k-d tree to rkyv file ({})\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}

fn serialize_to_rkyv(file: &mut File, tree: Tree) {
    let tree_rk: ImmutableKdTreeRK<f32, u32, 3, 64> = tree.into();

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
        .serialize_value(&tree_rk)
        .expect("Could not serialize with rkyv");

    let buf = serializer.into_serializer().into_inner();

    file.write_all(buf)
        .expect("Could not write serialized rkyv to file");
}
