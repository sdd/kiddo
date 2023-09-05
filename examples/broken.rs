use std::error::Error;

use kiddo::float_leaf_simd::leaf_node::LeafNode;
use rand::{SeedableRng, Rng};
use tracing_subscriber;


fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    // let tree_size = 8;
    // let seed = 0;

    // let tree_size = 18;
    // let seed = 894771;

    // let tree_size = 21;
    // let seed = 131851;

    let tree_size = 2usize.pow(27); // ~128M
    let seed: u64 = 1;//31851;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    // let content_to_add: Vec<[f64; 4]> = (0..tree_size).map(|_| rng.gen::<[f64; 4]>()).collect();

    // let mut duped: Vec<[f64; 4]> = Vec::with_capacity(content_to_add.len() * 10);
    // for item in content_to_add {
    //     for _ in 0..1{//6 {
    //         duped.push(item);
    //     }
    // }

    // let tree: ImmutableKdTree<f64, usize, 4, 64> = ImmutableKdTree::new_from_slice(&duped);

    // println!("Tree Stats: {:?}", tree.generate_stats());

    let simdleaf: LeafNode<f32, usize, 4, 32> = LeafNode {
        content_points: rng.gen::<[[f32; 32]; 4]>(),
        content_items: rng.gen::<[usize; 32]>(),
        size: 32
    };

    let mut best_dist = f32::INFINITY;
    let mut best_item = usize::MAX;

    simdleaf.nearest_one(&[0f32, 0f32, 0f32, 0f32], &mut best_dist, &mut best_item);

    Ok(())
}
