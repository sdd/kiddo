use az::Az;
use az::Cast;

// #[cfg(target_feature = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::f64_avx2::get_best_from_dists_f64_avx2;

#[cfg(target_feature = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::f64_avx512::get_best_from_dists_f64_avx512;

use super::fallback::get_best_from_dists_autovec;

use crate::{float::kdtree::Axis, types::Content};

#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    pub content_points: [[A; B]; K],
    pub content_items: [T; B],
    pub size: usize,
}

pub trait BestFromDists<T, const B: usize> {
    fn get_best_from_dists(acc: [Self; B], items: &[T; B], best_dist: &mut Self, best_item: &mut T)
    where
        Self: Sized;
}

impl<A, T, const K: usize, const B: usize> LeafNode<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    pub fn new() -> Self {
        LeafNode {
            content_items: [T::zero(); B],
            content_points: [[A::zero(); B]; K],
            size: 0,
        }
    }

    #[inline(never)]
    pub fn nearest_one(self: &Self, query: &[A; K], best_dist: &mut A, best_item: &mut T) {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [A::zero(); B];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; B];

            (0..B).step_by(1).for_each(|idx| {
                acc[idx] = acc[idx]
                    + (self.content_points[dim][idx] - qd[idx])
                        * (self.content_points[dim][idx] - qd[idx]);
            });
        });

        A::get_best_from_dists(acc, &self.content_items, best_dist, best_item);

        let (leaf_best_dist, leaf_best_item) =
            acc.iter()
                .enumerate()
                .fold((*best_dist, usize::MAX), |(bd, bi), (i, &d)| {
                    (
                        bd.min(d),
                        bi * usize::from(bd >= d) + i * usize::from(bd < d),
                    )
                });

        if leaf_best_dist < *best_dist {
            *best_dist = leaf_best_dist;
            *best_item = leaf_best_item.az::<T>();
        }
    }
}

impl<T: Content, const B: usize> BestFromDists<T, B> for f64
where
    T: Content,
    usize: Cast<T>,
{
    fn get_best_from_dists(acc: [f64; B], items: &[T; B], best_dist: &mut f64, best_item: &mut T) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                #[cfg(target_feature = "avx512f")]
                get_best_from_dists_f64_avx512(&acc, best_dist, best_item)
            } else if is_x86_feature_detected!("avx2") {
                // #[cfg(target_feature = "avx2")]
                unsafe { get_best_from_dists_f64_avx2(&acc, items, best_dist, best_item) }
            } else {
                get_best_from_dists_autovec(&acc, items, best_dist, best_item)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            get_best_from_dists_autovec(&acc, best_dist, best_item)
        }
    }
}

impl<T: Content, const B: usize> BestFromDists<T, B> for f32
where
    T: Content,
    usize: Cast<T>,
{
    fn get_best_from_dists(acc: [f32; B], items: &[T; B], best_dist: &mut f32, best_item: &mut T) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                // TODO
                unimplemented!()
            } else if is_x86_feature_detected!("avx2") {
                // TODO
                unimplemented!()
            } else {
                get_best_from_dists_autovec(&acc, items, best_dist, best_item)
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            get_best_from_dists_autovec(&acc, items, best_dist, best_item)
        }
    }
}
