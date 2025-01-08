use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use std::error::Error;
use std::fs::File;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;

use rkyv_08::{from_bytes_unchecked, rancor::Error as RkyvError};

// use kiddo::immutable::float::kdtree::ArchivedImmutableKdTree;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

type Tree = ImmutableKdTree<f64, u32, 3, 256>;

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
        unsafe { MmapOptions::new().map(&File::open("./examples/immutable-test-tree-r08.rkyv")?)? };

    // TODO: unsatisfied trait bounds when trying to call nearest_one on archived tree

    // safe API
    // let archived_tree = rkyv_08::access::<ArchivedImmutableKdTree<f64, u32, 3, 256>, RkyvError>(&buf[..]).unwrap();

    // faster unsafe API
    // let archived_tree =
    //     unsafe { rkyv_08::access_unchecked::<ArchivedImmutableKdTree<f64, u32, 3, 256>>(&buf) };

    // perform a query
    // let nearest_neighbour = archived_tree.nearest_one::<SquaredEuclidean>(&query);

    // println!(
    //     "total elapsed: {}\n\n",
    //     ElapsedDuration::new(start.elapsed())
    // );
    // println!(
    //     "Nearest item to query (archived): {:?}",
    //     nearest_neighbour.item
    // );

    // full deserialization
    let tree = unsafe { from_bytes_unchecked::<Tree, RkyvError>(&buf) }?;

    // perform a query
    let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

    println!(
        "Nearest item to query (deserialized): {:?}",
        nearest_neighbour.item
    );
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
