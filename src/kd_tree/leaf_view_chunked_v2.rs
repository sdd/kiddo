use crate::dist::{DistanceMetricCore, DistanceMetricUnified};
use crate::kd_tree::leaf_view::{try_identity_widen_axis, LeafView};
use crate::traits_unified_2::{
    AxisUnified, Basics, DistanceMetricUnified as DistanceMetricUnifiedV2,
};

/// Scalar nearest-one leaf processing using the V3 distance metric trait stack.
///
/// This is a direct leaf kernel (no orchestrator traversal logic) and is intended
/// as the baseline/fallback path for V3 integration.
#[inline(always)]
pub(crate) fn nearest_one_scalar_with_query_wide<AX, T, D, const K: usize, const B: usize>(
    leaf: LeafView<'_, AX, T, K, B>,
    query_wide: &[<D as DistanceMetricCore<AX>>::Output; K],
    best_dist: &mut <D as DistanceMetricCore<AX>>::Output,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX>,
    <D as DistanceMetricCore<AX>>::Output:
        AxisUnified<Coord = <D as DistanceMetricCore<AX>>::Output> + 'static,
{
    let (points, items) = leaf.into_parts();
    let len = items.len();
    if len == 0 {
        return;
    }

    type O<Dm, A> = <Dm as DistanceMetricCore<A>>::Output;
    let widened_points = points.map(try_identity_widen_axis::<AX, O<D, AX>>);

    if widened_points.iter().all(Option::is_some) {
        let widened_points = widened_points.map(|axis| axis.expect("checked is_some above"));
        for idx in 0..len {
            let mut dist = O::<D, AX>::zero();
            for dim in 0..K {
                unsafe {
                    dist += <D as DistanceMetricCore<AX>>::dist1(
                        *widened_points.get_unchecked(dim).get_unchecked(idx),
                        *query_wide.get_unchecked(dim),
                    );
                }
            }

            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { *items.get_unchecked(idx) };
            }
        }
        return;
    }

    for idx in 0..len {
        let mut dist = O::<D, AX>::zero();
        for dim in 0..K {
            unsafe {
                let coord = <D as DistanceMetricCore<AX>>::widen_coord(
                    *points.get_unchecked(dim).get_unchecked(idx),
                );
                dist += <D as DistanceMetricCore<AX>>::dist1(coord, *query_wide.get_unchecked(dim));
            }
        }

        if dist < *best_dist {
            *best_dist = dist;
            *best_item = unsafe { *items.get_unchecked(idx) };
        }
    }
}

/// Scalar nearest-one leaf processing entrypoint compatible with the existing
/// V2 `DistanceMetricUnified` query path.
#[inline(always)]
pub(crate) fn nearest_one_scalar_with_query_wide_v2<AX, T, D, const K: usize, const B: usize>(
    leaf: LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnifiedV2<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    let (points, items) = leaf.into_parts();
    let len = items.len();
    if len == 0 {
        return;
    }

    let widened_points = points.map(try_identity_widen_axis::<AX, D::Output>);

    if widened_points.iter().all(Option::is_some) {
        let widened_points = widened_points.map(|axis| axis.expect("checked is_some above"));
        for idx in 0..len {
            let mut dist = D::Output::zero();
            for dim in 0..K {
                unsafe {
                    dist += D::dist1(
                        *widened_points.get_unchecked(dim).get_unchecked(idx),
                        *query_wide.get_unchecked(dim),
                    );
                }
            }

            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { *items.get_unchecked(idx) };
            }
        }
        return;
    }

    for idx in 0..len {
        let mut dist = D::Output::zero();
        for dim in 0..K {
            unsafe {
                let coord = D::widen_coord(*points.get_unchecked(dim).get_unchecked(idx));
                dist += D::dist1(coord, *query_wide.get_unchecked(dim));
            }
        }

        if dist < *best_dist {
            *best_dist = dist;
            *best_item = unsafe { *items.get_unchecked(idx) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::nearest_one_scalar_with_query_wide;
    use crate::dist::{DistanceMetricCore, Manhattan, SquaredEuclidean};
    use crate::kd_tree::leaf_view::LeafView;

    #[test]
    fn nearest_one_scalar_v3_identity_f64_squared_euclidean() {
        let x = [0.0, 2.0, 5.0, -1.0];
        let y = [0.0, 1.0, 3.0, 7.0];
        let items = [10u32, 11u32, 12u32, 13u32];

        let leaf = LeafView::<f64, u32, 2, 32>::new([&x, &y], &items);

        type D = SquaredEuclidean<f64>;
        let query_wide = [1.8f64, 1.2f64];
        let mut best_dist = f64::INFINITY;
        let mut best_item = 0u32;

        nearest_one_scalar_with_query_wide::<f64, u32, D, 2, 32>(
            leaf,
            &query_wide,
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_item, 11);
        assert!((best_dist - 0.08).abs() < 1e-12);
    }

    #[test]
    fn nearest_one_scalar_v3_widen_f32_to_f64_squared_euclidean() {
        let x = [0.0f32, 2.0, 5.0, -1.0];
        let y = [0.0f32, 1.0, 3.0, 7.0];
        let items = [20u32, 21u32, 22u32, 23u32];

        let leaf = LeafView::<f32, u32, 2, 32>::new([&x, &y], &items);

        type D = SquaredEuclidean<f64>;
        let query = [1.8f32, 1.2f32];
        let query_wide = query.map(D::widen_coord);

        let mut best_dist = f64::INFINITY;
        let mut best_item = 0u32;

        nearest_one_scalar_with_query_wide::<f32, u32, D, 2, 32>(
            leaf,
            &query_wide,
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_item, 21);
        assert!((best_dist - 0.08).abs() < 1e-6);
    }

    #[test]
    fn nearest_one_scalar_v3_does_not_replace_when_not_better() {
        let x = [0.0f64, 2.0, 5.0];
        let y = [0.0f64, 1.0, 3.0];
        let items = [30u32, 31u32, 32u32];

        let leaf = LeafView::<f64, u32, 2, 32>::new([&x, &y], &items);

        type D = Manhattan<f64>;
        let query_wide = [2.0f64, 1.0f64];
        let mut best_dist = 0.0f64;
        let mut best_item = 999u32;

        nearest_one_scalar_with_query_wide::<f64, u32, D, 2, 32>(
            leaf,
            &query_wide,
            &mut best_dist,
            &mut best_item,
        );

        assert_eq!(best_item, 999);
        assert_eq!(best_dist, 0.0);
    }
}
