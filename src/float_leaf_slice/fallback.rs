use az::Cast;

use crate::{float::kdtree::Axis, types::Content};

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
