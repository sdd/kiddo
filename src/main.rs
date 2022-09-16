use sok::simd::f32::d4::distance::squared_euclidean;
use sok::simd::f32::d4::kdtree::KdTree;

fn rand_data_4d() -> ([f32; 4], usize) {
    rand::random()
}

fn main() {
    let points_to_add: Vec<([f32; 4], usize)> =
        (0..100).into_iter().map(|_| rand_data_4d()).collect();

    let mut kdtree = KdTree::with_capacity(points_to_add.len());

    for i in 0..points_to_add.len() {
        kdtree.add(&points_to_add[i].0, i);
        kdtree.nearest_one(&points_to_add[0].0, &squared_euclidean);
    }
}
