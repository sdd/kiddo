use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use std::error::Error;
use std::fs::File;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

type Tree = ImmutableKdTree<f64, 3>;

fn main() -> Result<(), Box<dyn Error>>
where
{
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    let query = [0.123f64, 0.456f64, 0.789f64];

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
    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!("Nearest item to query: {:?}", nearest_neighbour.item);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
