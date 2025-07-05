use elapsed::ElapsedDuration;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;
use ubyte::ToByteUnit;

use kiddo::distance::float::SquaredEuclidean;
use kiddo::mutable::float::kdtree::KdTree;

const NUM_ITEMS: usize = 50_000;

type Tree = KdTree<f64, u32, 3, 32, u32>;

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
    let tree: Tree = Tree::from_iter(
        content_to_add
            .into_iter()
            .enumerate()
            .map(|(idx, pt)| (pt, idx as u32)),
    );
    println!(
        "Populated KdTree with {} items ({})",
        tree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    let start = Instant::now();

    let buf = to_bytes::<RkyvError>(&tree)?;

    let mut file = File::create("./examples/float-test-tree-rkyv_08.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    let file_size = file.metadata().unwrap().len().bytes();
    println!(
        "Serialized k-d tree to rkyv file 'float-test-tree-rkyv_08.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    Ok(())
}
