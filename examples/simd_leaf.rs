use elapsed::ElapsedDuration;
use init_with::InitWith;
use num_traits::Zero;
use std::error::Error;
use std::time::Instant;

use kiddo::float::kdtree::LeafNode;
use kiddo::float_leaf_simd::leaf_node::LeafNode as SimdLeafNode;

use kiddo::distance_metric::DistanceMetric;
use kiddo::float::distance::SquaredEuclidean;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use tracing_subscriber;

type AX = f32;
const K: usize = 4;
const NUM_LEAVES: usize = 2usize.pow(21); // 2M
const BUCKET_SIZE: usize = 32;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let seed: u64 = 1; //31851;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let mut points: Vec<[AX; K]> = Vec::with_capacity(BUCKET_SIZE);

    let mut classic_leaves: Vec<_> = Vec::with_capacity(NUM_LEAVES);
    let mut simd_leaves: Vec<_> = Vec::with_capacity(NUM_LEAVES);

    for leaf_idx in 0..NUM_LEAVES {
        points.clear();
        let points = rng.gen::<[[AX; K]; BUCKET_SIZE]>();
        let items = <[usize; BUCKET_SIZE]>::init_with_indices(|i| leaf_idx * BUCKET_SIZE + i);

        let mut simd_points = [[AX::zero(); BUCKET_SIZE]; K];
        for dim in 0..K {
            for idx in 0..BUCKET_SIZE {
                simd_points[dim][idx] = points[idx][dim];
            }
        }

        let simd_leaf: SimdLeafNode<AX, usize, 4, BUCKET_SIZE> = SimdLeafNode {
            content_points: simd_points,
            content_items: items.clone(),
            size: BUCKET_SIZE,
        };

        let classic_leaf: LeafNode<AX, usize, 4, BUCKET_SIZE, u32> = LeafNode {
            content_points: points,
            content_items: items,
            size: BUCKET_SIZE as u32,
        };

        classic_leaves.push(classic_leaf);
        simd_leaves.push(simd_leaf);
    }

    let mut traverse_seq: Vec<usize> = (0..NUM_LEAVES).collect();
    traverse_seq.shuffle(&mut rng);

    let query = rng.gen::<[AX; K]>();

    let mut best_dist_classic = AX::MAX;
    let mut best_idx_classic = usize::MAX;

    let mut best_dist_simd = AX::MAX;
    let mut best_idx_simd = usize::MAX;

    let start = Instant::now();
    for leaf_idx in &traverse_seq {
        let leaf_node = unsafe { classic_leaves.get_unchecked(*leaf_idx) };

        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size as usize)
            .for_each(|(idx, entry)| {
                let dist = SquaredEuclidean::dist(&query, entry);
                if dist < best_dist_classic {
                    best_dist_classic = dist;
                    best_idx_classic = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                    // nearest.item = leaf_node.content_items[idx]
                }
            });
    }
    println!(
        "Searched classic leaves: best point = {:?}, best dist = {:?} ({})",
        best_idx_classic,
        best_dist_classic,
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    for leaf_idx in &traverse_seq {
        let leaf_node = unsafe { simd_leaves.get_unchecked(*leaf_idx) };

        leaf_node.nearest_one::<SquaredEuclidean>(&query, &mut best_dist_simd, &mut best_idx_simd)
    }
    println!(
        "Searched simd leaves: best point = {:?}, best dist = {:?} ({})",
        best_idx_classic,
        best_dist_classic,
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
