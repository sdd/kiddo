use az::Cast;
use std::collections::BinaryHeap;

use crate::float::result_collection::ResultCollection;
use crate::{float::kdtree::Axis, types::Content, BestNeighbour, NearestNeighbour};

#[inline]
pub(crate) fn get_best_from_dists_autovec<A: Axis, T: Content>(
    acc: &[A],
    items: &[T],
    best_dist: &mut A,
    best_item: &mut T,
) where
    usize: Cast<T>,
{
    // Autovectorizes with 256bit vectors on x86_64 where available
    // 341 loops (1 item per loop, unrolled x 3) of 4-8 instructions per item
    let (leaf_best_item, leaf_best_dist) = acc
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

#[inline]
pub(crate) fn update_nearest_dists_within_autovec<A: Axis, T: Content, R>(
    dists: &[A],
    items: &[T],
    radius: A,
    results: &mut R,
) where
    usize: Cast<T>,
    R: ResultCollection<A, T>,
{
    // TODO: Optimise with Godbolt
    dists
        .iter()
        .enumerate()
        .filter(|(_, &distance)| distance < radius)
        .for_each(|(idx, &distance)| {
            results.add(NearestNeighbour {
                distance,
                item: *unsafe { items.get_unchecked(idx) },
            });
        });
}

#[inline]
pub(crate) fn update_best_dists_within_autovec<A: Axis, T: Content>(
    dists: &[A],
    items: &[T],
    radius: A,
    max_qty: usize,
    results: &mut BinaryHeap<BestNeighbour<A, T>>,
) where
    usize: Cast<T>,
{
    // TODO: Optimise with Godbolt
    dists
        .iter()
        .enumerate()
        .filter(|(_, &distance)| distance <= radius)
        .for_each(|(idx, &distance)| {
            let item = *unsafe { items.get_unchecked(idx) };
            if results.len() < max_qty {
                results.push(BestNeighbour { distance, item });
            } else {
                let mut top = results.peek_mut().unwrap();
                if item < top.item {
                    top.item = item;
                    top.distance = distance;
                }
            }
        });
}
