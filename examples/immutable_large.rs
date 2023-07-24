use std::error::Error;

use kiddo::immutable_float::kdtree::ImmutableKdTree;
use rand::{SeedableRng, Rng};

const TREE_SIZE: usize = 2usize.pow(23); // ~8M


fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
    let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

    let tree: ImmutableKdTree<f32, usize, 4, 32> = ImmutableKdTree::optimize_from(&content_to_add);

    println!("Tree Stats: {:?}", tree.generate_stats());

    Ok(())
}
