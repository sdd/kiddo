use az::Cast;
use std::collections::BinaryHeap;

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
use crate::float::result_collection::ResultCollection;
use crate::{float::kdtree::Axis, types::Content, BestNeighbour, NearestNeighbour};

#[derive(Clone, Debug, PartialEq)]
pub struct PointSlice<'a, A: Copy + Default, T: Copy + Default, const K: usize> {
    points: [&'a [A]; K],
    items: &'a [T],
    len: usize,
}

pub trait BestFromDists<T: Content> {
    fn nearest_one(acc: &[Self], items: &[T], best_dist: &mut Self, best_item: &mut T)
    where
        Self: Axis;

    fn nearest_n_within<R: ResultCollection<Self, T>>(
        acc: &[Self],
        items: &[T],
        radius: Self,
        nearest: &mut R,
    ) where
        Self: Axis;

    fn best_n_within(
        acc: &[Self],
        items: &[T],
        max_qty: usize,
        radius: Self,
        nearest: &mut BinaryHeap<BestNeighbour<Self, T>>,
    ) where
        Self: Axis;
}

impl<'a, A, T, const K: usize> PointSlice<'a, A, T, K>
where
    A: Axis + BestFromDists<T>,
    T: Content,
    usize: Cast<T>,
{
    pub fn new(points: [&'a [A]; K], items: &'a [T]) -> Self {
        let len = points[0].len();
        debug_assert_eq!(len, items.len());
        PointSlice { points, items, len }
    }

    pub fn calc_dists<D>(&self, query: &[A; K]) -> Vec<A>
    where
        D: DistanceMetric<A, K>,
    {
        // TODO: use an array of length B as a window to avoid the allocations

        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = vec![A::zero(); self.len];
        (0..K).step_by(1).for_each(|dim| {
            let qd = vec![query[dim]; self.len];

            (0..self.len).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(self.points[dim][idx], qd[idx]);
            });
        });

        acc
    }

    pub fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        let dists = self.calc_dists::<D>(query);
        A::nearest_one(&dists, self.items, best_dist, best_item);
    }

    pub fn nearest_n_within<D, R>(&self, query: &[A; K], radius: A, nearest: &mut R)
    where
        R: ResultCollection<A, T>,
        D: DistanceMetric<A, K>,
    {
        let dists = self.calc_dists::<D>(query);
        A::nearest_n_within::<R>(&dists, self.items, radius, nearest);
    }

    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        max_qty: usize,
        radius: A,
        best: &mut BinaryHeap<BestNeighbour<A, T>>,
    ) where
        D: DistanceMetric<A, K>,
    {
        let dists = self.calc_dists::<D>(query);
        A::best_n_within(&dists, self.items, max_qty, radius, best);
    }
}

impl<T: Content> BestFromDists<T> for f64
where
    T: Content,
    usize: Cast<T>,
{
    fn nearest_one(acc: &[f64], items: &[T], best_dist: &mut f64, best_item: &mut T) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /*if is_x86_feature_detected!("avx512f") {
                #[cfg(target_feature = "avx512f")]
                unsafe {
                    get_best_from_dists_f64_avx512(acc, items, best_dist, best_item)
                }
            } else */
            if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f64_avx2(acc, items, best_dist, best_item)
                }
            } else {
                get_best_from_dists_autovec(acc, items, best_dist, best_item)
            }
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            get_best_from_dists_autovec(acc, items, best_dist, best_item)
        }
    }

    fn nearest_n_within<R>(acc: &[Self], items: &[T], radius: Self, nearest: &mut R)
    where
        Self: Sized,
        R: ResultCollection<Self, T>,
    {
        acc.iter().enumerate().for_each(|(idx, &distance)| {
            if distance < radius {
                nearest.add(NearestNeighbour {
                    distance,
                    item: *unsafe { items.get_unchecked(idx) },
                });
            }
        });
    }

    fn best_n_within(
        acc: &[Self],
        items: &[T],
        max_qty: usize,
        radius: Self,
        best: &mut BinaryHeap<BestNeighbour<Self, T>>,
    ) where
        Self: Sized,
    {
        acc.iter()
            .enumerate()
            .filter(|(_, &distance)| distance <= radius)
            .for_each(|(idx, &distance)| {
                let item = *unsafe { items.get_unchecked(idx) };
                if best.len() < max_qty {
                    best.push(BestNeighbour { distance, item });
                } else {
                    let mut top = best.peek_mut().unwrap();
                    if item < top.item {
                        top.item = item;
                        top.distance = distance;
                    }
                }
            });
    }
}

impl<T: Content> BestFromDists<T> for f32
where
    T: Content,
    usize: Cast<T>,
{
    fn nearest_one(acc: &[f32], items: &[T], best_dist: &mut f32, best_item: &mut T) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            /* if is_x86_feature_detected!("avx512f") {
                // TODO
                unimplemented!()
            } else */
            /*if is_x86_feature_detected!("avx2") {
                #[cfg(target_feature = "avx2")]
                unsafe {
                    get_best_from_dists_f32_avx2(acc, items, best_dist, best_item)
                }
            } else {*/
            get_best_from_dists_autovec(acc, items, best_dist, best_item)
            //}
        }

        #[cfg(any(
            not(feature = "simd"),
            not(any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            get_best_from_dists_autovec(acc, items, best_dist, best_item)
        }
    }

    fn nearest_n_within<R>(acc: &[Self], items: &[T], radius: Self, nearest: &mut R)
    where
        Self: Sized,
        R: ResultCollection<Self, T>,
    {
        acc.iter().enumerate().for_each(|(idx, &distance)| {
            if distance < radius {
                nearest.add(NearestNeighbour {
                    distance,
                    item: *unsafe { items.get_unchecked(idx) },
                });
            }
        });
    }

    fn best_n_within(
        acc: &[Self],
        items: &[T],
        max_qty: usize,
        radius: Self,
        best: &mut BinaryHeap<BestNeighbour<Self, T>>,
    ) where
        Self: Sized,
    {
        acc.iter()
            .enumerate()
            .filter(|(_, &distance)| distance <= radius)
            .for_each(|(idx, &distance)| {
                let item = *unsafe { items.get_unchecked(idx) };
                if best.len() < max_qty {
                    best.push(BestNeighbour { distance, item });
                } else {
                    let mut top = best.peek_mut().unwrap();
                    if item < top.item {
                        top.item = item;
                        top.distance = distance;
                    }
                }
            });
    }
}
