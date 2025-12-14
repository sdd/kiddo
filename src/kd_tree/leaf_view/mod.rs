use crate::kd_tree::result_collection::ResultCollection;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};
use crate::{BestNeighbour, NearestNeighbour};
use std::array;
use std::collections::BinaryHeap;

// TODO: chunking
#[allow(unused)]
const CHUNK_SIZE: usize = 32;

/// A view into a leaf node's data.
///
/// Provides a unified interface for accessing leaf data regardless of the underlying
/// storage strategy.
#[derive(Debug)]
pub struct LeafView<'a, AX, T, const K: usize, const B: usize> {
    points: [&'a [AX]; K],
    items: &'a [T],
}

impl<'a, AX: AxisUnified<Coord = AX>, T: Basics, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    pub(crate) fn new(points: [&'a [AX]; K], items: &'a [T]) -> Self {
        Self { points, items }
    }

    #[allow(dead_code)]
    pub(crate) fn into_parts(self) -> ([&'a [AX]; K], &'a [T]) {
        (self.points, self.items)
    }

    #[allow(dead_code)]
    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn len(&self) -> usize {
        self.points[0].len()
    }

    #[inline]
    pub(crate) fn nearest_one<D>(
        &self,
        query: &[AX; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: DistanceMetricUnified<AX, K>,
    {
        let dists = self.dists_for_slice::<D>(query);
        Self::update_nearest_dist(dists.as_slice(), self.items, best_dist, best_item);
    }

    #[inline]
    pub(crate) fn nearest_n_within<D, R>(
        &self,
        query: &[AX; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut R,
    ) where
        D: DistanceMetricUnified<AX, K>,
        R: ResultCollection<D::Output, T>,
    {
        let dists = self.dists_for_slice::<D>(query);
        Self::update_nearest_dists(dists.as_slice(), self.items, dist, results);
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dists_for_slice<D>(&self, query: &[AX; K]) -> Vec<D::Output>
    where
        D: DistanceMetricUnified<AX, K>,
    {
        let n = self.len();

        // accumulator of widened distance type
        let mut acc: Vec<D::Output> = vec![D::Output::zero(); n];

        // Pre-widen query coords once for this leaf
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut coord_wide: [Vec<D::Output>; K] = array::from_fn(|_| vec![D::Output::zero(); n]);

        (0..K).for_each(|dim| {
            (0..n).for_each(|idx| {
                coord_wide[dim][idx] = D::widen_coord(self.points[dim][idx]);
            });
        });

        (0..K).for_each(|dim| {
            (0..n).for_each(|idx| {
                acc[idx] += D::dist1(coord_wide[dim][idx], query_wide[dim]);
            });
        });

        acc
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dist<O>(
        dists: &[O],
        items: &[T],
        best_dist: &mut O,
        best_item: &mut T,
    ) where
        O: AxisUnified<Coord = O>,
    {
        // Autovectorizes with 256bit vectors on x86_64 where available
        // 341 loops (1 item per loop, unrolled x 3) of 4-8 instructions per item
        let (leaf_best_item, leaf_best_dist) = dists
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // 6 instructions, 1 branch
        if *leaf_best_dist < *best_dist {
            *best_dist = *leaf_best_dist;
            *best_item = items[leaf_best_item];
        }
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_nearest_dists<O, R>(dists: &[O], items: &[T], dist: O, results: &mut R)
    where
        O: AxisUnified<Coord = O>,
        R: ResultCollection<O, T>,
    {
        dists.iter().zip(items).for_each(|(&d, &i)| {
            if d < dist {
                results.add(NearestNeighbour {
                    distance: d,
                    item: i,
                });
            }
        })
    }
}

impl<'a, AX: AxisUnified<Coord = AX>, T: Basics + Ord, const K: usize, const B: usize>
    LeafView<'a, AX, T, K, B>
{
    #[inline]
    pub(crate) fn best_n_within<D>(
        &self,
        query: &[AX; K],
        dist: <D as DistanceMetricUnified<AX, K>>::Output,
        results: &mut BinaryHeap<BestNeighbour<<D as DistanceMetricUnified<AX, K>>::Output, T>>,
    ) where
        D: DistanceMetricUnified<AX, K>,
    {
        let dists = self.dists_for_slice::<D>(query);
        Self::update_best_dists(
            dists.as_slice(),
            self.items,
            dist,
            results.capacity(),
            results,
        );
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    pub(crate) fn update_best_dists<O>(
        dists: &[O],
        items: &[T],
        dist: O,
        max_qty: usize,
        results: &mut BinaryHeap<BestNeighbour<O, T>>,
    ) where
        O: AxisUnified<Coord = O>,
    {
        dists.iter().zip(items).for_each(|(&d, &item)| {
            if d <= dist {
                if results.len() < max_qty {
                    results.push(BestNeighbour { distance: d, item });
                } else {
                    let mut top = results.peek_mut().unwrap();
                    if item < top.item {
                        top.item = item;
                        top.distance = d;
                    }
                }
            }
        })
    }
}
