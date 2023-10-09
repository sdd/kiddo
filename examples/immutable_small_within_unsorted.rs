// use elapsed::ElapsedDuration;
// use kiddo::immutable::float::kdtree::ImmutableKdTree;
// use kiddo::test_utils::build_query_points_float;
// use rand::Rng;
use std::error::Error;
// use std::time::Instant;

// use criterion::black_box;
// use elapsed::ElapsedDuration;
// use kiddo::float::distance::SquaredEuclidean;
// use rand::{Rng, SeedableRng};
// use rayon::iter::IntoParallelRefIterator;
// use rayon::iter::ParallelIterator;
// use std::time::Instant;
//
// use kiddo::immutable::float::kdtree::ImmutableKdTree;
// use kiddo::test_utils::build_query_points_float;
//
// const TREE_SIZE: usize = 2usize.pow(23); // ~8M
// const QUERY_POINT_QTY: usize = 10_000;

fn main() -> Result<(), Box<dyn Error>> {
    /*let mut rng = rand_chacha::ChaCha8Rng::from_entropy();
    let content_to_add: Vec<[f64; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f64; 4]>()).collect();

    let start = Instant::now();
    println!("Building an optimized tree of {:?} items...", TREE_SIZE);
    let tree: ImmutableKdTree<f64, usize, 4, 32> = ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    println!("Tree Stats: {:?}", tree.generate_stats());

    let query_points = build_query_points_float(QUERY_POINT_QTY);
    println!(
        "Performing {:?} random within_unsorted queries...",
        QUERY_POINT_QTY
    );

    let start = Instant::now();
    black_box({
        query_points.par_iter().for_each(|point| {
            black_box({
                tree.within_unsorted::<SquaredEuclidean>(point, 0.01);
            })
        });
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );*/

    Ok(())
}
