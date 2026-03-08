use crate::kd_tree::leaf_view::{try_identity_widen_axis, LeafView};
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};
use array_init::array_init;
use std::mem::MaybeUninit;

const CHUNK_SIZE: usize = 32;

#[inline(always)]
pub(crate) fn try_nearest_one_with_query_wide<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    let points = leaf.points();
    let widened_points = points.map(|axis| try_identity_widen_axis::<AX, D::Output>(axis));

    if widened_points.iter().any(Option::is_none) {
        return false;
    }

    let widened_points = widened_points.map(|axis| axis.unwrap());
    let items = leaf.items();
    let mut points_iterators = widened_points.map(|axis| axis.chunks_exact(CHUNK_SIZE));
    let mut items_iterator = items.chunks_exact(CHUNK_SIZE);

    while let Some(items_chunk_slice) = items_iterator.next() {
        let points_chunk: [&[D::Output; CHUNK_SIZE]; K] = array_init(|dim| {
            points_iterators[dim]
                .next()
                .expect("points/items chunk mismatch")
                .try_into()
                .expect("invalid points chunk length")
        });

        let items_chunk: &[T; CHUNK_SIZE] = items_chunk_slice
            .try_into()
            .expect("invalid items chunk length");

        let dists = dists_for_chunk::<AX, D, K, CHUNK_SIZE>(points_chunk, query_wide);
        update_nearest_dist_chunk(dists, items_chunk, best_dist, best_item);
    }

    let remainder_points: [&[D::Output]; K] = array_init(|dim| points_iterators[dim].remainder());
    let remainder_items = items_iterator.remainder();

    // TODO: For FlatVec specifically, we can experiment with overreading the tail into the next
    // contiguous leaf (or padded end-of-vec region) to keep the distance kernel branchless, while
    // masking/scalarizing only the final min-selection step. Keep this first pass leaf-bounded so
    // correctness is easy to validate and benchmark against the current scalar fallback.
    if !remainder_items.is_empty() {
        for idx in 0..remainder_items.len() {
            let mut dist = D::Output::zero();
            for dim in 0..K {
                unsafe {
                    dist += D::dist1(
                        *remainder_points.get_unchecked(dim).get_unchecked(idx),
                        *query_wide.get_unchecked(dim),
                    );
                }
            }

            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { *remainder_items.get_unchecked(idx) };
            }
        }
    }

    true
}

#[inline(always)]
fn dists_for_chunk<A, D, const K: usize, const C: usize>(
    chunk: [&[D::Output; C]; K],
    query: &[D::Output; K],
) -> [D::Output; C]
where
    A: Copy,
    D: DistanceMetricUnified<A, K>,
{
    let mut acc = [const { MaybeUninit::<D::Output>::uninit() }; C];
    let acc_ptr = acc.as_mut_ptr() as *mut D::Output;

    let q = unsafe { *query.get_unchecked(0) };
    let axis = unsafe { *chunk.get_unchecked(0) };

    for idx in 0..C {
        unsafe {
            *acc_ptr.add(idx) = D::dist1(*axis.get_unchecked(idx), q);
        }
    }

    for dim in 1..K {
        let q = unsafe { *query.get_unchecked(dim) };
        let axis = unsafe { *chunk.get_unchecked(dim) };

        for idx in 0..C {
            unsafe {
                *acc_ptr.add(idx) += D::dist1(*axis.get_unchecked(idx), q);
            }
        }
    }

    unsafe { std::ptr::read(acc.as_ptr() as *const [D::Output; C]) }
}

#[inline(always)]
fn update_nearest_dist_chunk<O, T, const C: usize>(
    acc: [O; C],
    items: &[T; C],
    best_dist: &mut O,
    best_item: &mut T,
) where
    O: AxisUnified<Coord = O>,
    T: Basics,
{
    let mut candidate_idx = 0usize;
    let mut candidate_dist = *best_dist;
    let mut improved = false;

    for idx in 0..C {
        let dist = unsafe { *acc.get_unchecked(idx) };
        if dist < candidate_dist {
            candidate_dist = dist;
            candidate_idx = idx;
            improved = true;
        }
    }

    if improved {
        *best_dist = candidate_dist;
        *best_item = unsafe { *items.get_unchecked(candidate_idx) };
    }
}
