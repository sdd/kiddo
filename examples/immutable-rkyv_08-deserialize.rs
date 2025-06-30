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

use rkyv::{access, access_unchecked, from_bytes_unchecked, rancor::Error as RkyvError};

use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
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
    let file = File::open("./examples/immutable-test-tree-rkyv_08.rkyv")?;
    let buf = unsafe { MmapOptions::new().map(&file)? };

    {
        // full deserialization
        let start = Instant::now();
        let tree = unsafe { from_bytes_unchecked::<Tree, RkyvError>(&buf) }?;
        let loaded = Instant::now();

        // perform some queries
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

        let pre_query = Instant::now();
        let approx_nearest_neighbour = tree.approx_nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Approx nearest item to query (deserialized): {:?}",
            approx_nearest_neighbour.item
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let dist = 0.01;
        let max_qty = NonZero::new(10).unwrap();

        let pre_query = Instant::now();
        let best_n_within = tree.best_n_within::<SquaredEuclidean>(&query, dist, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        let best_n_within = best_n_within.collect::<Vec<_>>();

        println!("Best n items within radius of query (deserialized): {best_n_within:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n = tree.nearest_n::<SquaredEuclidean>(&query, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        println!("Nearest n items of query (deserialized): {nearest_n:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n_within =
            tree.nearest_n_within::<SquaredEuclidean>(&query, dist, max_qty, true);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "Nearest n items (sorted) within radius of query (deserialized): {nearest_n_within:?}"
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within = tree.within::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, sorted (deserialized): ({:?} items)",
            within.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within_unsorted = tree.within_unsorted::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, unsorted (deserialized): ({:?} items)",
            within_unsorted.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));
    }

    {
        // Safe mode Zero Copy Deserialization
        let start = Instant::now();

        // Get archived tree
        let archived_tree =
            access::<ArchivedR8ImmutableKdTree<f64, u32, 3, 256>, RkyvError>(&buf[..]).unwrap();
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

        let pre_query = Instant::now();
        let approx_nearest_neighbour = archived_tree.approx_nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Approx nearest item to query (checked ZC): {:?}",
            approx_nearest_neighbour.item
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let dist = 0.01;
        let max_qty = NonZero::new(10).unwrap();

        let pre_query = Instant::now();
        let best_n_within = archived_tree.best_n_within::<SquaredEuclidean>(&query, dist, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        let best_n_within = best_n_within.collect::<Vec<_>>();

        println!("Best n items within radius of query (checked ZC): {best_n_within:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n = archived_tree.nearest_n::<SquaredEuclidean>(&query, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        println!("Nearest n items of query (checked ZC): {nearest_n:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n_within =
            archived_tree.nearest_n_within::<SquaredEuclidean>(&query, dist, max_qty, true);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "Nearest n items (sorted) within radius of query (checked ZC): {nearest_n_within:?}"
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within = archived_tree.within::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, sorted (checked ZC): ({:?} items)",
            within.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within_unsorted = archived_tree.within_unsorted::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, unsorted (checked ZC): ({:?} items)",
            within_unsorted.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));
    }

    {
        // faster unsafe ZC Deserialize API
        let start = Instant::now();

        // Get archived tree using unsafe method
        let archived_tree =
            unsafe { access_unchecked::<ArchivedR8ImmutableKdTree<f64, u32, 3, 256>>(&buf) };
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

        let pre_query = Instant::now();
        let approx_nearest_neighbour = archived_tree.approx_nearest_one::<SquaredEuclidean>(&query);

        println!(
            "Approx nearest item to query (unchecked ZC): {:?}",
            approx_nearest_neighbour.item
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let dist = 0.01;
        let max_qty = NonZero::new(10).unwrap();

        let pre_query = Instant::now();
        let best_n_within = archived_tree.best_n_within::<SquaredEuclidean>(&query, dist, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        let best_n_within = best_n_within.collect::<Vec<_>>();

        println!("Best n items within radius of query (unchecked ZC): {best_n_within:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n = archived_tree.nearest_n::<SquaredEuclidean>(&query, max_qty);
        ElapsedDuration::new(pre_query.elapsed());

        println!("Nearest n items of query (unchecked ZC): {nearest_n:?}");
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let nearest_n_within =
            archived_tree.nearest_n_within::<SquaredEuclidean>(&query, dist, max_qty, true);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "Nearest n items (sorted) within radius of query (unchecked ZC): {nearest_n_within:?}"
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within = archived_tree.within::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, sorted (unchecked ZC): ({:?} items)",
            within.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));

        let pre_query = Instant::now();
        let within_unsorted = archived_tree.within_unsorted::<SquaredEuclidean>(&query, dist);
        ElapsedDuration::new(pre_query.elapsed());

        println!(
            "All items within radius of query, unsorted (unchecked ZC): ({:?} items)",
            within_unsorted.len()
        );
        println!("took {}.\n", ElapsedDuration::new(pre_query.elapsed()));
    }

    Ok(())
}
