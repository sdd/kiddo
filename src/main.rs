// use sok::tuned::f32::d4::distance::squared_euclidean;
// use sok::tuned::f32::d4::kdtree::KdTree;

use fixed::types::extra::U14;
use fixed::FixedU16;
use sok::tuned::u16::d4::distance::squared_euclidean;
use sok::tuned::u16::d4::kdtree::KdTree;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;

// fn rand_data_4d_f32() -> ([f32; K], usize) { rand::random() }
fn rand_data_4d_u16() -> ([u16; K], u32) { rand::random() }

// fn main() {
//     let points_to_add: Vec<([f32; K], usize)> =
//         (0..100).into_iter().map(|_| rand_data_4d_f32()).collect();
//
//     let mut kdtree = KdTree::with_capacity(points_to_add.len());
//
//     for i in 0..points_to_add.len() {
//         kdtree.add(&points_to_add[i].0, i);
//         kdtree.nearest_one(&points_to_add[0].0, &squared_euclidean);
//     }
// }


fn main() {
    let points_to_add: Vec<([FixedU16<U14>; K], u32)> =
        (0..100).into_iter()
            .map(|_| rand_data_4d_u16())
            .map(|(p, i)| (unsafe { std::mem::transmute(p) }, i))
            .collect();

    let mut kdtree = KdTree::with_capacity(points_to_add.len());

    for i in 0..points_to_add.len() {
        kdtree.add(&points_to_add[i].0, i as u32);
        kdtree.nearest_one(&points_to_add[0].0, &squared_euclidean);
    }
}
