use std::error::Error;
use std::fs::File;
use std::num::NonZero;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::kd_tree::{ArchivedKdTree, KdTree};
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::EytzingerPf;
use kiddo::SquaredEuclidean;
use memmap::MmapOptions;
use rkyv_08::api::high::from_bytes;
use rkyv_08::rancor::Error as RkyvError;
use rkyv_08::{access, access_unchecked};
#[cfg(feature = "tracing")]
use tracing::Level;
#[cfg(feature = "tracing")]
use tracing_subscriber::fmt;

type Tree = KdTree<f64, u32, EytzingerPf<3, 8>, VecOfArenas<f64, u32, 3, 256>, 3, 256>;
type ArchivedTree =
    ArchivedKdTree<f64, u32, EytzingerPf<3, 8>, VecOfArenas<f64, u32, 3, 256>, 3, 256>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::TRACE).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    let query = [0.123f64, 0.456f64, 0.789f64];
    let file = File::open("./examples/immutable-test-tree-rkyv_08.rkyv")?;
    let buf = unsafe { MmapOptions::new().map(&file)? };

    let start = Instant::now();
    let owned_tree = from_bytes::<Tree, RkyvError>(&buf[..])?;
    let loaded = Instant::now();
    let nearest_neighbour = owned_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    println!(
        "Nearest item to query (owned from_bytes): {:?}",
        nearest_neighbour.1
    );
    println!(
        "took {} total, {} loading.\n",
        ElapsedDuration::new(start.elapsed()),
        ElapsedDuration::new(loaded - start)
    );

    let start = Instant::now();
    let archived_tree = access::<ArchivedTree, RkyvError>(&buf[..])?;
    let loaded = Instant::now();
    println!(
        "Checked zero-copy metadata: size={} leaf_count={} max_stem_level={} max_leaf_len={}",
        archived_tree.size(),
        archived_tree.leaf_count(),
        archived_tree.max_stem_level(),
        archived_tree.max_leaf_len()
    );

    let nearest_neighbour = archived_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    println!(
        "Nearest item to query (checked zero-copy archived): {:?}",
        nearest_neighbour.1
    );
    println!(
        "took {} total, {} access.\n",
        ElapsedDuration::new(start.elapsed()),
        ElapsedDuration::new(loaded - start)
    );

    let approx_nearest_neighbour = archived_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .approx()
        .execute();
    println!(
        "Approx nearest item to query (zero-copy archived): {:?}",
        approx_nearest_neighbour.1
    );

    let dist = 0.01;
    let max_qty = NonZero::new(10usize).unwrap();

    let best_n_within = archived_tree
        .query(&query)
        .best_n_within::<SquaredEuclidean<f64>>(dist, max_qty)
        .execute()
        .into_sorted_vec();
    println!("Best n items within radius of query: {best_n_within:?}");

    let nearest_n = archived_tree
        .query(&query)
        .nearest_n::<SquaredEuclidean<f64>>(max_qty)
        .execute();
    println!("Nearest n items: {nearest_n:?}");

    let nearest_n_within = archived_tree
        .query(&query)
        .nearest_n::<SquaredEuclidean<f64>>(max_qty)
        .within(dist)
        .execute();
    println!("Nearest n items within radius: {nearest_n_within:?}");

    let within = archived_tree
        .query(&query)
        .within::<SquaredEuclidean<f64>>(dist)
        .execute();
    println!("All items within radius, sorted: {} items", within.len());

    let within_unsorted = archived_tree
        .query(&query)
        .within::<SquaredEuclidean<f64>>(dist)
        .unsorted()
        .execute();
    println!(
        "All items within radius, unsorted: {} items",
        within_unsorted.len()
    );

    let archived_tree = unsafe { access_unchecked::<ArchivedTree>(&buf) };
    println!(
        "Unchecked zero-copy metadata: size={} leaf_count={} max_stem_level={} max_leaf_len={}",
        archived_tree.size(),
        archived_tree.leaf_count(),
        archived_tree.max_stem_level(),
        archived_tree.max_leaf_len()
    );

    let start = Instant::now();
    let nearest_neighbour = archived_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f64>>()
        .execute();
    println!(
        "Nearest item to query (unchecked zero-copy archived): {:?}",
        nearest_neighbour.1
    );
    println!("took {}.\n", ElapsedDuration::new(start.elapsed()));

    Ok(())
}
