use sok::KdTree;

const BUCKET_SIZE: usize = 30;

fn rand_data_4d() -> ([f64; 4], usize) {
    rand::random()
}

fn main() {
    let points_to_add: Vec<([f64; 4], usize)> =
        (0..100).into_iter().map(|_| rand_data_4d()).collect();

    let mut kdtree =
        KdTree::<f64, usize, 4, BUCKET_SIZE>::with_capacity(points_to_add.len());

    for i in 0..points_to_add.len() {
        kdtree.add(&points_to_add[i].0, i);
    }
}
