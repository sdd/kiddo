use crate::sok::Axis;

//#[inline(never)]
pub(crate) fn distance_to_bounds<F, A, const K: usize>(
    p1: &[A; K],
    bounds: &[(A, A); K],
    distance: &F,
) -> A
where
    F: Fn(&[A; K], &[A; K]) -> A,
    A: Axis,
{
    let mut p2 = [A::nan(); K];

    //// SLOWER
    // for i in 0..K {
    //     p2[i] = clamp(p1[i], bounds[i]);
    // }

    p1.iter()
        .zip(bounds)
        .map(|(&v, &bounds)| clamp(v, bounds))
        .zip(p2.iter_mut())
        .for_each(|(clamped_val, p2_coord)| *p2_coord = clamped_val);

    distance(p1, &p2)
}

//#[inline(never)]
pub(crate) fn clamp<A: Axis>(val: A, bounds: (A, A)) -> A {
    if val < bounds.0 {
        bounds.0
    } else if val > bounds.1 {
        bounds.1
    } else {
        val
    }
}

//#[inline(never)]
pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
    bounds.iter_mut().enumerate().for_each(|(dim, bound)| {
        if point[dim] < bound.0 {
            bound.0 = point[dim];
        }
        if point[dim] > bound.1 {
            bound.1 = point[dim];
        }
    });
}

//// SLOWER
// #[inline(never)]
// pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
//     point.iter().zip(bounds.iter_mut()).for_each(|(&point_coord, bound)| {
//         if point_coord < bound.0 {
//             bound.0 = point_coord;
//         }
//         if point_coord > bound.1 {
//             bound.1 = point_coord;
//         }
//     });
// }

//// SLOWER
// #[inline(never)]
// pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
//     point.iter().zip(bounds.iter_mut()).for_each(|(&point_coord, bound)| {
//         bound.0 = bound.0.min(point_coord);
//         bound.1 = bound.1.max(point_coord);
//     });
// }

//// DOESN'T WORK
// #[inline(never)]
// pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
//     bounds = point.iter().zip(bounds.iter()).map(|(&point_coord, (&lo, &hi))| {
//         (lo.min(point_coord), hi.max(point_coord))
//     }).collect();
// }
