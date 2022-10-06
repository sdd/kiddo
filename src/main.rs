use sok::tuned::u16::dn::distance::manhattan;
use sok::tuned::u16::dn::kdtree::KdTree;

use fixed::types::extra::U16;
use fixed::FixedU16;

type FXD = FixedU16<U16>;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;

fn n<const K: usize>(pt: [f32; K]) -> [FXD; K] {
    pt.map(|num| FXD::from_num(num))
}

//fn rand_data<A, const K: usize>() -> ([A; K], u32) { rand::random() }

fn rand_data_4d() -> ([f32; 4], u32) {
    rand::random()
}


/*fn main() {
    let points_to_add: Vec<([f32; K], usize)> =
        (0..100).into_iter().map(|_| rand_data_4d_f32()).collect();

    let mut kdtree = KdTree::with_capacity(points_to_add.len());

    for i in 0..points_to_add.len() {
        kdtree.add(&points_to_add[i].0, i);
        kdtree.nearest_one(&points_to_add[0].0, &squared_euclidean);
    }
}*/

fn main() {
    let points_to_add: Vec<([FXD; K], u32)> =
        (0..100).into_iter()
            .map(|_| rand_data_4d())
            .map(|(p, i)| (unsafe { std::mem::transmute(n(p)) }, i))
            .collect();

    let mut kdtree: KdTree<FXD, u32, K, BUCKET_SIZE, u32> = KdTree::with_capacity(points_to_add.len());

    for i in 0..points_to_add.len() {
        kdtree.add(&points_to_add[i].0, i as u32);
        kdtree.nearest_one(&points_to_add[0].0, &manhattan);
    }
}
