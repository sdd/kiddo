use az::Cast;
use std::slice::ChunksExact;

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
#[derive(Debug)]
pub struct LeafFixedSlice<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    pub content_points: [&'a [A; C]; K],
    pub content_items: &'a [T; C],
}

impl<A, T, const K: usize, const C: usize> LeafFixedSlice<'_, A, T, K, C>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    pub fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [A::zero(); C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];

            (0..C).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(self.content_points[dim][idx], qd[idx]);
            });
        });

        A::update_best_dist(acc, self.content_items, best_dist, best_item);
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct LeafSlice<'a, A: Axis, T: Content, const K: usize> {
    pub content_points: [&'a [A]; K],
    pub content_items: &'a [T],
}

impl<A: Axis, T: Content, const K: usize> LeafSlice<'_, A, T, K> {
    #[allow(dead_code)]
    #[inline]
    fn len(&self) -> usize {
        self.content_items.len()
    }
}

pub struct LeafFixedSliceIterator<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    points_iterators: [ChunksExact<'a, A>; K],
    items_iterator: ChunksExact<'a, T>,
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize> Iterator
    for LeafFixedSliceIterator<'a, A, T, K, C>
{
    type Item = ([&'a [A; C]; K], &'a [T; C]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.items_iterator.is_empty() {
            None
        } else {
            let points_chunk: [&[A; C]; K] = array_init::array_init(|i| {
                self.points_iterators[i].next().unwrap().try_into().unwrap()
            });
            let item_chunk: &[T; C] = self.items_iterator.next().unwrap().try_into().unwrap();
            Some((points_chunk, item_chunk))
        }
    }
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize>
    LeafFixedSliceIterator<'a, A, T, K, C>
{
    fn remainder(&self) -> ([&'a [A]; K], &'a [T]) {
        (
            array_init::array_init(|i| self.points_iterators[i].remainder()),
            self.items_iterator.remainder(),
        )
    }
}

pub trait LeafSliceFloat<T, const K: usize> {
    fn update_best_dist<const C: usize>(
        acc: [Self; C],
        items: &[T; C],
        best_dist: &mut Self,
        best_item: &mut T,
    ) where
        Self: Sized;

    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized;
}

impl<A, T, const K: usize> LeafSlice<'_, A, T, K>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    pub fn new<'a>(content_points: [&'a [A]; K], content_items: &'a [T]) -> LeafSlice<'a, A, T, K> {
        let size = content_items.len();
        for arr in content_points {
            debug_assert_eq!(arr.len(), size);
        }

        LeafSlice {
            content_items,
            content_points,
        }
    }

    fn as_full_chunks<const C: usize>(&self) -> LeafFixedSliceIterator<A, T, K, C> {
        let points_iterators = self.content_points.map(|i| i.chunks_exact(C));
        let items_iterator = self.content_items.chunks_exact(C);

        LeafFixedSliceIterator {
            items_iterator,
            points_iterators,
        }
    }

    pub fn nearest_one<D>(&self, query: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        let chunk_iter = self.as_full_chunks::<64>();
        let (remainder_points, remainder_items) = chunk_iter.remainder();
        for chunk in chunk_iter {
            let dists = A::dists_for_chunk::<D, 64>(chunk.0, query);
            A::update_best_dist(dists, chunk.1, best_dist, best_item);
        }

        #[allow(clippy::needless_range_loop)]
        for idx in 0..remainder_items.len() {
            let mut dist = A::zero();
            (0..K).step_by(1).for_each(|dim| {
                dist += D::dist1(remainder_points[dim][idx], query[dim]);
            });

            // TODO: make branchless
            let dist_is_better = u8::from(dist < *best_dist);
            // best_dist.cmovnz(&dist, dist_is_better);
            // best_item.cmovnz(&remainder_items[idx], dist_is_better);

            if dist_is_better == 1 {
                *best_dist = dist;
                *best_item = remainder_items[idx];
            }
        }
    }
}

impl<T: Content, const K: usize> LeafSliceFloat<T, K> for f64
where
    T: Content,
    usize: Cast<T>,
{
    fn update_best_dist<const C: usize>(
        acc: [f64; C],
        items: &[T; C],
        best_dist: &mut f64,
        best_item: &mut T,
    ) {
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

    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [0f64; C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];

            (0..C).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(chunk[dim][idx], qd[idx]);
            });
        });

        acc
    }
}

impl<T: Content, const K: usize> LeafSliceFloat<T, K> for f32
where
    T: Content,
    usize: Cast<T>,
{
    fn update_best_dist<const C: usize>(
        acc: [f32; C],
        items: &[T; C],
        best_dist: &mut f32,
        best_item: &mut T,
    ) {
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

    fn dists_for_chunk<D, const C: usize>(chunk: [&[Self; C]; K], query: &[Self; K]) -> [Self; C]
    where
        D: DistanceMetric<Self, K>,
        Self: Sized,
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [0f32; C];
        (0..K).step_by(1).for_each(|dim| {
            let qd = [query[dim]; C];

            (0..C).step_by(1).for_each(|idx| {
                acc[idx] += D::dist1(chunk[dim][idx], qd[idx]);
            });
        });

        acc
    }
}
