use std::error::Error;

use criterion::black_box;
use elapsed::ElapsedDuration;
use kiddo::float::distance::squared_euclidean;
use rand::{Rng, SeedableRng};
// use rayon::iter::IntoParallelRefIterator;
// use rayon::iter::ParallelIterator;
use std::time::Instant;

// use kiddo::float::kdtree::KdTree;
use kiddo::immutable_float::kdtree::ImmutableKdTree;
use kiddo::test_utils::build_query_points_float;

/*
   Tree Construction Times:

   Tree Size        Time
   2^20             0.15 - 1s
   2^21             1.3 - 4.9s
   2^22             13.7 - 31s
   2^23             92 - 98s
*/

// const TREE_SIZE: usize = 2usize.pow(23); // ~8M
const TREE_SIZE: usize = 2usize.pow(21); // ~2M
const QUERY_POINT_QTY: usize = 10_000_000;

fn main() -> Result<(), Box<dyn Error>> {
    // let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
    let mut rng = rand_chacha::ChaCha8Rng::from_entropy();
    let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

    let start = Instant::now();
    println!("Building an optimized tree of {:?} items...", TREE_SIZE);
    let tree: ImmutableKdTree<f32, usize, 4, 32> = ImmutableKdTree::optimize_from(&content_to_add);
    // let tree: KdTree<f32, u32, 4, 32, u32> = KdTree::from(&content_to_add);
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // println!("Tree Stats: {:?}", tree.generate_stats());

    let query_points = build_query_points_float(QUERY_POINT_QTY);
    println!("Performing {:?} random NN queries...", QUERY_POINT_QTY);

    let start = Instant::now();
    black_box({
        query_points.iter().for_each(|point| {
            black_box({
                tree.nearest_one(point, &squared_euclidean);
            })
        });
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
