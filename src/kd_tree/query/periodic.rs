use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;

use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, Content, LeafStrategy, QueryResultItem, StemStrategy};

use super::builder::{boundary_accepts, BoundaryMode, PeriodicAxis, QueryBuilderTreeOps};

fn with_wrapped_queries<A: PeriodicAxis, F, const K: usize>(
    query: &[A; K],
    box_size: &[A; K],
    mut f: F,
) where
    F: FnMut(&[A; K]),
{
    fn recurse<A: PeriodicAxis, F, const K: usize>(
        query: &[A; K],
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        f: &mut F,
    ) where
        F: FnMut(&[A; K]),
    {
        if axis == K {
            f(wrapped_query);
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        wrapped_query[axis] = original - axis_len;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        wrapped_query[axis] = original;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        let mut plus = original;
        plus += axis_len;
        wrapped_query[axis] = plus;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        wrapped_query[axis] = original;
    }

    let mut wrapped_query = *query;
    recurse(query, box_size, 0, &mut wrapped_query, &mut f);
}

#[inline(always)]
fn periodic_image_axis_offset<A: PeriodicAxis>(wrapped_coord: A, axis_len: A) -> A {
    if A::cmp(wrapped_coord, A::zero()) == Ordering::Less {
        A::saturating_dist(wrapped_coord, A::zero())
    } else if A::cmp(wrapped_coord, axis_len) == Ordering::Greater {
        A::saturating_dist(wrapped_coord, axis_len)
    } else {
        A::zero()
    }
}

#[derive(Clone, Copy)]
struct PeriodicImageCandidate<A, O, const K: usize> {
    wrapped_query: [A; K],
    lower_bound: O,
}

fn periodic_image_candidates<D, A, const K: usize>(
    query: &[A; K],
    box_size: &[A; K],
    threshold: Option<(BoundaryMode, D::Output)>,
    include_home: bool,
) -> Vec<PeriodicImageCandidate<A, D::Output, K>>
where
    A: PeriodicAxis + 'static,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
{
    fn recurse<D, A, const K: usize>(
        query: &[A; K],
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        has_non_zero_shift: bool,
        lower_bound: D::Output,
        threshold: Option<(BoundaryMode, D::Output)>,
        include_home: bool,
        out: &mut Vec<PeriodicImageCandidate<A, D::Output, K>>,
    ) where
        A: PeriodicAxis + 'static,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: Axis<Coord = D::Output>,
    {
        if axis == K {
            if include_home || has_non_zero_shift {
                out.push(PeriodicImageCandidate {
                    wrapped_query: *wrapped_query,
                    lower_bound,
                });
            }
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        for shift in [-1_i8, 0, 1] {
            let wrapped_coord = match shift {
                -1 => original - axis_len,
                0 => original,
                1 => {
                    let mut plus = original;
                    plus += axis_len;
                    plus
                }
                _ => unreachable!(),
            };

            wrapped_query[axis] = wrapped_coord;

            let offset = periodic_image_axis_offset(wrapped_coord, axis_len);
            let mut next_lower_bound = lower_bound;
            D::combine_component(
                &mut next_lower_bound,
                D::dist1(D::widen_coord(offset), D::Output::zero()),
            );

            if threshold.is_some_and(|(boundary, limit)| {
                !boundary_accepts(boundary, next_lower_bound, limit)
            }) {
                continue;
            }

            recurse::<D, A, K>(
                query,
                box_size,
                axis + 1,
                wrapped_query,
                has_non_zero_shift || shift != 0,
                next_lower_bound,
                threshold,
                include_home,
                out,
            );
        }

        wrapped_query[axis] = original;
    }

    let mut candidates = Vec::new();
    let mut wrapped_query = *query;
    recurse::<D, A, K>(
        query,
        box_size,
        0,
        &mut wrapped_query,
        false,
        D::Output::zero(),
        threshold,
        include_home,
        &mut candidates,
    );
    candidates
}

#[inline(always)]
fn sort_periodic_image_candidates<A, O, const K: usize>(
    candidates: &mut [PeriodicImageCandidate<A, O, K>],
) where
    O: Axis<Coord = O>,
{
    candidates.sort_unstable_by(|lhs, rhs| {
        lhs.lower_bound
            .partial_cmp(&rhs.lower_bound)
            .unwrap_or(Ordering::Equal)
    });
}

pub(crate) fn periodic_nearest_one_result<Tree, A, T, SS, LS, D, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
) -> QueryResultItem<(), T, D::Output>
where
    A: PeriodicAxis + 'static,
    T: Content,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
{
    assert!(
        A::periodic_box_is_valid(box_size),
        "periodic box sizes must be strictly positive"
    );

    if D::ORDERING != Ordering::Less {
        let mut best_result = QueryResultItem {
            point: (),
            item: T::default(),
            distance: D::Output::max_value(),
        };

        with_wrapped_queries::<A, _, K>(query, box_size, |wrapped_query| {
            let (distance, item) = tree.qb_nearest_one::<D>(wrapped_query);
            if D::Output::cmp(distance, best_result.distance) == Ordering::Less {
                best_result.distance = distance;
                best_result.item = item;
            }
        });

        return best_result;
    }

    let (home_distance, home_item) = tree.qb_nearest_one::<D>(query);
    let mut best_result = QueryResultItem {
        point: (),
        item: home_item,
        distance: home_distance,
    };

    let mut candidates = periodic_image_candidates::<D, A, K>(
        query,
        box_size,
        Some((BoundaryMode::Exclusive, best_result.distance)),
        false,
    );
    sort_periodic_image_candidates(&mut candidates);

    for candidate in candidates {
        if !boundary_accepts(
            BoundaryMode::Exclusive,
            candidate.lower_bound,
            best_result.distance,
        ) {
            continue;
        }

        let (distance, item) = tree.qb_nearest_one::<D>(&candidate.wrapped_query);
        if D::Output::cmp(distance, best_result.distance) == Ordering::Less {
            best_result.distance = distance;
            best_result.item = item;
        }
    }

    best_result
}

pub(crate) fn periodic_nearest_results_by_item<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<NonZeroUsize>,
    sorted: bool,
) -> Vec<QueryResultItem<(), T, D::Output>>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    assert!(
        A::periodic_box_is_valid(box_size),
        "periodic box sizes must be strictly positive"
    );

    if D::ORDERING != Ordering::Less {
        let mut best_by_item = HashMap::<T, D::Output>::new();

        with_wrapped_queries::<A, _, K>(query, box_size, |wrapped_query| {
            let candidates = match (radius, max_qty) {
                (None, Some(max_qty)) => tree.qb_nearest_n::<D>(wrapped_query, max_qty, true),
                (Some(radius), Some(max_qty)) => {
                    tree.qb_nearest_n_within::<D, EXCLUSIVE>(wrapped_query, radius, max_qty, sorted)
                }
                (Some(radius), None) if sorted => {
                    tree.qb_within::<D, EXCLUSIVE>(wrapped_query, radius)
                }
                (Some(radius), None) => {
                    tree.qb_within_unsorted::<D, EXCLUSIVE>(wrapped_query, radius)
                }
                (None, None) => {
                    unreachable!("periodic plural queries require radius and/or max_qty")
                }
            };

            for candidate in candidates {
                best_by_item
                    .entry(candidate.item)
                    .and_modify(|best_distance| {
                        if D::Output::cmp(candidate.distance, *best_distance) == Ordering::Less {
                            *best_distance = candidate.distance;
                        }
                    })
                    .or_insert(candidate.distance);
            }
        });

        let mut results: Vec<_> = best_by_item
            .into_iter()
            .map(|(item, distance)| QueryResultItem {
                point: (),
                item,
                distance,
            })
            .collect();

        if sorted {
            results.sort_unstable();
        }
        if let Some(max_qty) = max_qty {
            results.truncate(max_qty.get());
        }

        return results;
    }

    let mut best_by_item = HashMap::<T, D::Output>::new();
    let mut threshold = radius.map(|value| {
        (
            if EXCLUSIVE {
                BoundaryMode::Exclusive
            } else {
                BoundaryMode::Inclusive
            },
            value,
        )
    });

    let mut candidates = periodic_image_candidates::<D, A, K>(query, box_size, threshold, true);
    sort_periodic_image_candidates(&mut candidates);

    for candidate in candidates {
        if threshold.is_some_and(|(boundary, limit)| {
            !boundary_accepts(boundary, candidate.lower_bound, limit)
        }) {
            continue;
        }

        let candidates = match (radius, max_qty) {
            (None, Some(max_qty)) => {
                tree.qb_nearest_n::<D>(&candidate.wrapped_query, max_qty, true)
            }
            (Some(radius), Some(max_qty)) => tree.qb_nearest_n_within::<D, EXCLUSIVE>(
                &candidate.wrapped_query,
                radius,
                max_qty,
                sorted,
            ),
            (Some(radius), None) if sorted => {
                tree.qb_within::<D, EXCLUSIVE>(&candidate.wrapped_query, radius)
            }
            (Some(radius), None) => {
                tree.qb_within_unsorted::<D, EXCLUSIVE>(&candidate.wrapped_query, radius)
            }
            (None, None) => unreachable!("periodic plural queries require radius and/or max_qty"),
        };

        for periodic_candidate in candidates {
            best_by_item
                .entry(periodic_candidate.item)
                .and_modify(|best_distance| {
                    if D::Output::cmp(periodic_candidate.distance, *best_distance) == Ordering::Less
                    {
                        *best_distance = periodic_candidate.distance;
                    }
                })
                .or_insert(periodic_candidate.distance);
        }

        if radius.is_none() {
            if max_qty.is_some_and(|max_qty| best_by_item.len() >= max_qty.get()) {
                if let Some(current_worst) = best_by_item
                    .values()
                    .copied()
                    .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
                {
                    threshold = Some((BoundaryMode::Exclusive, current_worst));
                }
            }
        } else if let Some(radius) = radius {
            if max_qty.is_none_or(|max_qty| best_by_item.len() < max_qty.get()) {
                continue;
            }

            if let Some(current_worst) = best_by_item
                .values()
                .copied()
                .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
            {
                let tightened = if D::Output::cmp(current_worst, radius) == Ordering::Less {
                    current_worst
                } else {
                    radius
                };
                threshold = Some((
                    if EXCLUSIVE {
                        BoundaryMode::Exclusive
                    } else {
                        BoundaryMode::Inclusive
                    },
                    tightened,
                ));
            }
        }
    }

    let mut results: Vec<_> = best_by_item
        .into_iter()
        .map(|(item, distance)| QueryResultItem {
            point: (),
            item,
            distance,
        })
        .collect();

    if sorted {
        results.sort_unstable();
    }
    if let Some(max_qty) = max_qty {
        results.truncate(max_qty.get());
    }

    results
}
