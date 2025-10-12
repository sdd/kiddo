use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use rkyv_08::access_unchecked;
use rkyv_08::vec::ArchivedVec;

use kiddo::cache_simulator::{profiles, AccessKind};
use kiddo::distance::float::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::stem_strategies::Donnelly;

const BUCKET_SIZE: usize = 2;

type Tree = ImmutableKdTree<f64, usize, Donnelly<3, 64, 8, 4>, 4, BUCKET_SIZE>;
type ArchivedTree = ArchivedR8ImmutableKdTree<f64, usize, Donnelly<3, 64, 8, 4>, 4, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    // faster unsafe ZC Deserialize API
    let start = Instant::now();

    // memmap the tree file into a buffer
    let tree_file = File::open("./examples/immutable-ann-test-tree-rkyv_08.rkyv")?;
    let tree_buf = unsafe { MmapOptions::new().map(&tree_file)? };

    // Get archived tree using unsafe method
    let tree = unsafe { access_unchecked::<ArchivedTree>(&tree_buf) };

    // memmap the tree file into a buffer
    let query_file = File::open("./examples/immutable-ann-test-points-rkyv_08.rkyv")?;
    let query_buf = unsafe { MmapOptions::new().map(&query_file)? };

    // Get archived tree using unsafe method
    let query_points = unsafe { access_unchecked::<ArchivedVec<[f64; 4]>>(&query_buf) };

    println!(
        "Deserialization complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    println!("Performing {:?} random NN queries...", query_points.len());

    let (tx, rx) = std::sync::mpsc::channel::<usize>();

    let simulator_thread = std::thread::spawn(move || {
        let mut sim = profiles::zen3();
        let mut count = 0;

        while let Ok(idx) = rx.recv() {
            sim.step(idx);
            count += 1;
            if count % 10000 == 0 {
                println!("{:?} steps complete.", count);
            }
        }
        println!("{:?} steps complete.", count);

        let s = sim.snapshot_stats();
        println!(
            "L1: {} hits, {} misses | L2: {} hits, {} misses | L3: {} hits, {} misses | MEM: {}",
            s.l1.hits,
            s.l1.misses,
            s.l2.hits,
            s.l2.misses,
            s.l3.hits,
            s.l3.misses,
            s.memory_accesses
        );

        // per-set heatmap
        let (_h, m, e, _) = sim.l1.per_set_stats();
        for (i, &misses) in m.iter().enumerate() {
            println!("set {:4} : {:6} misses, {:6} evictions", i, misses, e[i]);
        }
    });

    let tx = Some(tx);
    let start = Instant::now();
    query_points.iter().for_each(|point| {
        black_box(tree.get_leaf_node_idx(point, tx.as_ref()));
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    drop(tx);

    simulator_thread.join().unwrap();

    Ok(())
}
