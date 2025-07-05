use az::Cast;
use std::collections::BinaryHeap;

use crate::mutable::float::result_collection::ResultCollection;
use crate::traits::{Axis, Content};
use crate::{BestNeighbour, NearestNeighbour};

#[inline]
pub(crate) fn update_nearest_dist_autovec<A: Axis, T: Content>(
    dists: &[A],
    items: &[T],
    best_dist: &mut A,
    best_item: &mut T,
) where
    usize: Cast<T>,
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
        .zip(items.iter())
        .filter(|(&distance, _)| distance <= radius)
        .for_each(|(&distance, &item)| {
            results.add(NearestNeighbour { distance, item });
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
        .zip(items.iter())
        .filter(|(&distance, _)| distance <= radius)
        .for_each(|(&distance, &item)| {
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

#[cfg(test)]
mod tests {
    use super::{
        update_best_dists_within_autovec, update_nearest_dist_autovec,
        update_nearest_dists_within_autovec,
    };
    use crate::{BestNeighbour, NearestNeighbour};
    use std::collections::BinaryHeap;

    #[test]
    fn test_get_best_from_dists_autovec_leaves_best_unchanged_when_not_better() {
        let dists = [10000f64, 20000f64, 20f64];
        let items = [1u32, 3u32, 5u32];

        let mut best_dist = 10f64;
        let mut best_item = 12345u32;

        update_nearest_dist_autovec(&dists[..], &items[..], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 10f64);
        assert_eq!(best_item, 12345u32);
    }

    #[test]
    fn test_get_best_from_dists_autovec_updates_best_when_closer_dist_present() {
        let dists = [10000f64, 20000f64, 2f64];
        let items = [1u32, 3u32, 5u32];

        let mut best_dist = 10f64;
        let mut best_item = 12345u32;

        update_nearest_dist_autovec(&dists[..], &items[..], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 2f64);
        assert_eq!(best_item, 5u32);
    }

    #[test]
    fn test_update_nearest_dists_within_autovec_leaves_nearest() {
        let dists = [10000f64, 20000f64, 20f64];
        let items = [1u32, 3u32, 5u32];

        let radius = 200f64;

        let mut results = vec![NearestNeighbour {
            distance: 10f64,
            item: 100u32,
        }];

        update_nearest_dists_within_autovec(&dists[..], &items[..], radius, &mut results);

        assert_eq!(
            results,
            vec![
                NearestNeighbour {
                    distance: 10f64,
                    item: 100u32
                },
                NearestNeighbour {
                    distance: 20f64,
                    item: 5u32
                }
            ]
        );
    }

    #[test]
    fn test_update_best_dists_within_autovec_leaves_nearest() {
        let dists = [10000f64, 20000f64, 20f64, 15f64];
        let items = [1u32, 3u32, 5u32, 7u32];

        let radius = 200f64;

        let max_qty = 2usize;

        let mut results = BinaryHeap::new();
        results.push(BestNeighbour {
            distance: 10f64,
            item: 100u32,
        });

        update_best_dists_within_autovec(&dists[..], &items[..], radius, max_qty, &mut results);

        let results = results.into_vec();

        assert_eq!(
            results,
            vec![
                BestNeighbour {
                    distance: 15f64,
                    item: 7u32
                },
                BestNeighbour {
                    distance: 20f64,
                    item: 5u32
                },
            ]
        );
    }
}
