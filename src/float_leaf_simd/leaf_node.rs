use az::Cast;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::f64_avx2::get_best_from_dists_f64_avx2;
//use super::{f32_avx2::get_best_from_dists_f32_avx2};

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// use super::f64_avx512::get_best_from_dists_f64_avx512;

use super::fallback::get_best_from_dists_autovec;

use crate::distance_metric::DistanceMetric;
use crate::{float::kdtree::Axis, types::Content};

#[doc(hidden)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    #[cfg_attr(
        feature = "serialize",
        serde(with = "crate::custom_serde::array_of_arrays")
    )]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))
    )]
    pub content_points: [[A; B]; K],

    #[cfg_attr(feature = "serialize", serde(with = "crate::custom_serde::array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
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

    pub fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [A::zero(); B];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; B];

            (0..B).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(self.content_points[dim][idx], qd[idx]);
            });
        });

        A::get_best_from_dists(acc, &self.content_items, best_dist, best_item);
    }
}

impl<A, T, const K: usize, const B: usize> Default for LeafNode<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rkyv")]
impl<A, T, const K: usize, const B: usize> ArchivedLeafNode<A, T, K, B>
where
    A: Axis + BestFromDists<T, B> + rkyv::Archive<Archived = A>,
    T: Content + rkyv::Archive<Archived = T>,
    usize: Cast<T>,
{
    pub fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [A::zero(); B];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; B];

            (0..B).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(self.content_points[dim][idx], qd[idx]);
            });
        });

        A::get_best_from_dists(acc, &self.content_items, best_dist, best_item);
    }
}

impl<T: Content, const B: usize> BestFromDists<T, B> for f64
where
    T: Content,
    usize: Cast<T>,
{
    fn get_best_from_dists(acc: [f64; B], items: &[T; B], best_dist: &mut f64, best_item: &mut T) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /*if is_x86_feature_detected!("avx512f") {
                #[cfg(target_feature = "avx512f")]
                unsafe {
                    get_best_from_dists_f64_avx512(&acc, items, best_dist, best_item)
                }
            } else */
            if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f64_avx2(&acc, items, best_dist, best_item)
                }
            } else {
                get_best_from_dists_autovec(&acc, items, best_dist, best_item)
            }
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            get_best_from_dists_autovec(&acc, items, best_dist, best_item)
        }
    }
}

impl<T: Content, const B: usize> BestFromDists<T, B> for f32
where
    T: Content,
    usize: Cast<T>,
{
    fn get_best_from_dists(acc: [f32; B], items: &[T; B], best_dist: &mut f32, best_item: &mut T) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /* if is_x86_feature_detected!("avx512f") {
                // TODO
                unimplemented!()
            } else */
            /*if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f32_avx2(&acc, items, best_dist, best_item)
                }
            } else {*/
            get_best_from_dists_autovec(&acc, items, best_dist, best_item)
            //}
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            get_best_from_dists_autovec(&acc, items, best_dist, best_item)
        }
    }
}
