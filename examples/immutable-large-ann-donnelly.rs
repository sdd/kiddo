use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::distance::float::SquaredEuclidean;
use rand::{Rng, SeedableRng};

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::stem_strategies::Donnelly;
use kiddo::test_utils::build_query_points_float;

const TREE_SIZE: usize = 2usize.pow(23);
const QUERY_POINT_QTY: usize = 20_000_000;
const BUCKET_SIZE: usize = 2;

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand_chacha::ChaCha8Rng::from_os_rng();
    let content_to_add: Vec<[f64; 4]> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

    let start = Instant::now();
    println!("Building an optimized tree of {TREE_SIZE:?} items...");
    let tree: ImmutableKdTree<f64, usize, Donnelly<3, 64, 8, 4>, 4, BUCKET_SIZE> =
        ImmutableKdTree::new_from_slice(&content_to_add);
    println!(
        "Construction complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // println!("Tree Stats: {:?}", tree.generate_stats());

    let query_points = build_query_points_float(QUERY_POINT_QTY);
    println!("Performing {QUERY_POINT_QTY:?} random NN queries...");

    let start = Instant::now();
    query_points.iter().for_each(|point| {
        black_box(tree.approx_nearest_one::<SquaredEuclidean>(point));
    });
    println!(
        "Queries complete. ({})",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
