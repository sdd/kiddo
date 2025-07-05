// This example builds a very large ImmutableTree with ~250M items,
// and then performs some random queries against it.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::distance::float::SquaredEuclidean;
use rand::{Rng, SeedableRng};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::test_utils::build_query_points_float;

const TREE_SIZE: usize = 2usize.pow(28); // ~250M
const QUERY_POINT_QTY: usize = 10_000_000;

/*
   Tree Construction Times (f64, Ryzen 5900X):

   Tree Size        Time
   2^20             0.25s
   2^21             0.7s
   2^22             1.7s
   2^23             4.3 - 4.5s
   2^24             10.2 - 10.7s
   2^25             23.5s
   2^26             54.3s
   2^27             138.2s
   2^28             351s (20gb RAM usage)
   2^29             (OOM)

   Tree Construction Times (f32, Ryzen 5900X):

   Tree Size        Time
   2^20             0.15 - 0.46s
   2^21             0.8 - 1.3s
   2^22             5.0 - 5.8s
   2^23             20 - 50s
*/

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand_chacha::ChaCha8Rng::from_os_rng();
    let content_to_add: Vec<[f64; 4]> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

    let start = Instant::now();
    println!("Building an optimized tree of {TREE_SIZE:?} items...");
    let tree: ImmutableKdTree<f64, usize, 4, 32> = ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // println!("Tree Stats: {:?}", tree.generate_stats());

    let query_points = build_query_points_float(QUERY_POINT_QTY);
    println!("Performing {QUERY_POINT_QTY:?} random NN queries...");

    let start = Instant::now();
    query_points.par_iter().for_each(|point| {
        black_box(tree.nearest_one::<SquaredEuclidean>(point));
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
