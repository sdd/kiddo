use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::cache_simulator::profiles;
use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
use kiddo::stem_strategies::Eytzinger;
use memmap::MmapOptions;
use rkyv_08::access_unchecked;
use rkyv_08::vec::ArchivedVec;

const BUCKET_SIZE: usize = 2;

type ArchivedTree = ArchivedR8ImmutableKdTree<f64, usize, Eytzinger<4>, 4, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    // faster unsafe ZC Deserialize API
    let start = Instant::now();

    // memmap the tree file into a buffer
    let tree_file = File::open("./examples/immutable-ann-test-tree-eytz-rkyv_08.rkyv")?;
    let tree_buf = unsafe { MmapOptions::new().map(&tree_file)? };

    // Get archived tree using unsafe method
    let tree = unsafe { access_unchecked::<ArchivedTree>(&tree_buf) };

    // memmap the tree file into a buffer
    let query_file = File::open("./examples/immutable-ann-test-points-eytz-rkyv_08.rkyv")?;
    let query_buf = unsafe { MmapOptions::new().map(&query_file)? };

    // Get archived tree using unsafe method
    let query_points = unsafe { access_unchecked::<ArchivedVec<[f64; 4]>>(&query_buf) };
    let total_queries = query_points.len();

    println!(
        "Deserialization complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    println!("Performing {:?} random NN queries...", query_points.len());

    let (tx, rx) = std::sync::mpsc::channel::<kiddo::cache_simulator::Event>();

    let simulator_thread = std::thread::spawn(move || {
        let mut sim = profiles::zen3();
        let mut count = 0u64;

        while let Ok(event) = rx.recv() {
            sim.step_event(event);

            count += 1;
            if count.is_multiple_of(10_000) {
                println!("{count} events processed...");
            }
        }

        println!("{count} events processed total.");

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

        // L1 set heatmap
        let (_h, m, e, _) = sim.l1.per_set_stats();
        for (i, &misses) in m.iter().enumerate() {
            println!(
                "set {i:4} : {misses:6} misses, {evictions:6} evictions",
                evictions = e[i]
            );
        }

        println!("Total simulated cycles: {}", sim.cycle);
        println!("Avg cycles per query: {}", sim.cycle / total_queries as u64);
        println!(
            "L1 prefetch useful: {}, late: {}",
            sim.pf_stats.l1.useful_lead_cycles_sum, sim.pf_stats.l1.late
        );

        println!("{}", sim.stride_analyzer.render_histogram(60));
        println!("{}", sim.stride_analyzer.render_markov(100));
        sim.print_top_addresses(100);
    });

    let start = Instant::now();
    query_points.iter().for_each(|point| {
        black_box(tree.simulate_traversal(point, &tx));
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    drop(tx);

    simulator_thread.join().unwrap();

    Ok(())
}
