use crate::sok::Axis;

// pub (crate) struct BoundsChecker<'a, F, A, const K: usize> where
//     F: Fn(&[A; K], &[A; K]) -> A,
//     A: Axis,
// {
//     pub(crate) min_bound: [A; K],
//     pub(crate) max_bound: [A; K],
//     pub(crate) distance_fn: &'a F,
// }
//
// impl<'a, 'b, F, A: Axis, const K: usize> BoundsChecker<'a, F, A, K> where
//     F: Fn(&[A; K], &[A; K]) -> A,
//     A: Axis,
// {
//     #[inline(never)]
//     pub(crate) fn new(
//         min_bound: [A; K],
//         max_bound: [A; K],
//         distance_fn: &'a F,
//     ) -> BoundsChecker<'a,F, A, K>
//     {
//         BoundsChecker {
//             min_bound,
//             max_bound,
//             distance_fn
//         }
//     }
// }

//
// impl<'a, 'b, F, A, const K: usize> BoundsChecker<'a, F, A, K> where
//     F: Fn(&[A; K], &[A; K]) -> A,
//     A: Axis
// {
//     #[inline(never)]
//     pub(crate) fn dist_to_bound(&self, p1: [A; K]) -> A
//     where
//     F: Fn(&[A; K], &[A; K]) -> A,
//     A: Axis,
//     {
//         let mut p2 = [A::nan(); K];
//
//         //// SLOWER
//         // let mut p2 = [A::nan(); K];
//         // for i in 0..K {
//         //     p2[i] = clamp(p1[i], bounds[i]);
//         // }
//
//         p1.into_iter()
//             .zip(self.min_bound)
//             .zip(&self.max_bound)
//             .map(|((v, min_bound), &max_bound)| clamp(v, min_bound, max_bound))
//             .zip(p2.iter_mut())
//             .for_each(|(clamped_val, p2_coord)| *p2_coord = clamped_val);
//
//         (self.distance_fn)(&p1, &p2)
//     }
// }



#[inline(never)]
pub(crate) fn distance_to_bounds<F, A, const K: usize>(
    p1: &[A; K],
    min_bound: &[A; K],
    max_bound: &[A; K],
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
        .zip(min_bound)
        .zip(max_bound)
        .map(|((&v, &min_bound), &max_bound)| clamp(v, min_bound, max_bound))
        .zip(p2.iter_mut())
        .for_each(|(clamped_val, p2_coord)| *p2_coord = clamped_val);

    distance(p1, &p2)
}

#[inline(never)]
pub(crate) fn distance_to_bounds_simd<F, A, const K: usize>(
    p1: &[A; K],
    min_bound: &[A; K],
    max_bound: &[A; K],
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
        .zip(min_bound)
        .zip(max_bound)
        .map(|((&v, &min_bound), &max_bound)| clamp(v, min_bound, max_bound))
        .zip(p2.iter_mut())
        .for_each(|(clamped_val, p2_coord)| *p2_coord = clamped_val);

    distance(p1, &p2)
}

#[inline(never)]
pub(crate) fn clamp<A: Axis>(val: A, min_bound: A, max_bound: A) -> A {
    if val < min_bound {
        min_bound
    } else if val > max_bound {
        max_bound
    } else {
        val
    }
}

#[inline(never)]
pub(crate) fn extend<A: Axis, const K: usize>(min_bound: &mut [A; K], max_bound: &mut [A; K], point: &[A; K]) {
    min_bound.iter_mut().enumerate().for_each(|(dim, bound)| {
        if point[dim] < *bound {
            *bound = point[dim];
        }
    });

    max_bound.iter_mut().enumerate().for_each(|(dim, bound)| {
        if point[dim] > *bound {
            *bound = point[dim];
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
