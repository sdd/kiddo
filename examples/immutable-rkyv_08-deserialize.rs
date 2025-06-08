use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use std::error::Error;
use std::fs::File;
use std::num::NonZero;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;

use rkyv_08::{access, access_unchecked, from_bytes_unchecked, rancor::Error as RkyvError};

use kiddo::immutable::float::kdtree::ArchivedImmutableKdTree;
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

    // memmap the file into a buffer
    let file = File::open("./examples/immutable-test-tree-r08.rkyv")?;
    let buf = unsafe { MmapOptions::new().map(&file)? };

    {
        // full deserialization
        let start = Instant::now();
        let tree = unsafe { from_bytes_unchecked::<Tree, RkyvError>(&buf) }?;
        let loaded = Instant::now();

        // perform a query
        let nearest_neighbour = tree.nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Nearest item to query (deserialized): {:?}",
            nearest_neighbour.item
        );
        println!(
            "took {} total, {} loading.\n\n",
            ElapsedDuration::new(start.elapsed()),
            ElapsedDuration::new(loaded - start)
        );
    }

    {
        // Safe mode Zero Copy Deserialization
        let start = Instant::now();

        // Get archived tree
        let archived_tree =
            access::<ArchivedImmutableKdTree<f64, u32, 3, 256>, RkyvError>(&buf[..]).unwrap();
        let loaded = Instant::now();

        // perform a query using the wrapper
        let nearest_neighbour = archived_tree.nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Nearest item to query (checked ZC): {:?}",
            nearest_neighbour.item
        );
        println!(
            "took {} total, {} loading.\n\n",
            ElapsedDuration::new(start.elapsed()),
            ElapsedDuration::new(loaded - start)
        );
    }

    {
        // faster unsafe ZC Deserialize API
        let start = Instant::now();

        // Get archived tree using unsafe method
        let archived_tree =
            unsafe { access_unchecked::<ArchivedImmutableKdTree<f64, u32, 3, 256>>(&buf) };
        let loaded = Instant::now();

        // perform a query using the wrapper
        let nearest_neighbour = archived_tree.nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Nearest item to query (unchecked ZC): {:?}",
            nearest_neighbour.item
        );
        println!(
            "took {} total, {} loading.\n\n",
            ElapsedDuration::new(start.elapsed()),
            ElapsedDuration::new(loaded - start)
        );

        let within = archived_tree.within::<SquaredEuclidean>(&query, 0.01);
        println!(
            "items within 0.01 of query (unchecked ZC): {:?}",
            within.len()
        );

        let best_n_within = archived_tree
            .best_n_within::<SquaredEuclidean>(&query, 0.01, NonZero::new(10).unwrap())
            .collect::<Vec<_>>();
        println!(
            "best 10 items within 0.01 of query (unchecked ZC): {:?}",
            best_n_within
        );
    }

    Ok(())
}
