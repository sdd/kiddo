use std::error::Error;

use kiddo::float_leaf_simd::leaf_node::LeafNode;

use rand::{Rng, SeedableRng};
use tracing_subscriber;

const LEAF_COUNT: usize = 100;
const QUERY_COUNT: usize = 1_000;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let seed: u64 = 1; //31851;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let mut leaves: Vec<_> = Vec::with_capacity(LEAF_COUNT);

    for _ in 0..LEAF_COUNT {
        let simdleaf: LeafNode<f32, usize, 4, 32> = LeafNode {
            content_points: rng.gen::<[[f32; 32]; 4]>(),
            content_items: rng.gen::<[usize; 32]>(),
            size: 32,
        };

        leaves.push(simdleaf);
    }

    let mut query_points = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        query_points.push(rng.gen::<[f32; 4]>());
    }

    for i in 0..QUERY_COUNT {
        let mut best_dist = f32::INFINITY;
        let mut best_item = usize::MAX;

        let leaf_idx = rng.gen_range(0..LEAF_COUNT);
        let leaf = &leaves[leaf_idx];

        let query_point = &query_points[i];

        let result = leaf.nearest_one(query_point, &mut best_dist, &mut best_item);
        println!("result: {:?}", &result);
    }
    Ok(())
}
