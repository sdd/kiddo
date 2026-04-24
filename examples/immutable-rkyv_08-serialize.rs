use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::kd_tree::KdTree;
use kiddo::stem_strategy::EytzingerPf;
use kiddo::SquaredEuclidean;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;
use ubyte::ToByteUnit;

const NUM_ITEMS: usize = 50_000_000;

type Tree = KdTree<f64, u32, EytzingerPf<3, 8>, VecOfArenas<f64, u32, 3, 256>, 3, 256>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    let query = [0.123f64, 0.456f64, 0.789f64];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let points: Vec<[f64; 3]> = (0..NUM_ITEMS).map(|_| rng.random::<[f64; 3]>()).collect();

    let start = Instant::now();
    let tree: Tree = KdTree::new_from_slice(&points);
    println!(
        "Populated KdTree with {} items ({})",
        tree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean<f64>>(&query);
    println!("Nearest item to query: {:?}", nearest_neighbour.1);

    let start = Instant::now();
    let buf = to_bytes::<RkyvError>(&tree)?;
    let mut file = File::create("./examples/immutable-test-tree-rkyv_08.rkyv")?;
    file.write_all(&buf)?;

    let file_size = file.metadata()?.len().bytes();
    println!(
        "Serialized k-d tree to 'immutable-test-tree-rkyv_08.rkyv' ({}). File size: {:.2}",
        ElapsedDuration::new(start.elapsed()),
        file_size
    );

    Ok(())
}
