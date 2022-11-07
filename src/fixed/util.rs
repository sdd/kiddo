use crate::fixed::kdtree::Axis;

#[allow(dead_code)]
pub(crate) fn distance_to_bounds<A: Axis, const K: usize, F>(p1: &[A; K], min_bound: &[A; K], max_bound: &[A; K], distance: &F) -> A
where
    F: Fn(&[A; K], &[A; K]) -> A,
{
    let mut p2 = [A::ZERO; K];

    p1.iter()
        .zip(min_bound)
        .zip(max_bound)
        .map(|((&v, &min_bound), &max_bound)| v.clamp(min_bound, max_bound))
        .zip(p2.iter_mut())
        .for_each(|(clamped_val, p2_coord)| *p2_coord = clamped_val);

    distance(p1, &p2)
}

// #[allow(dead_code)]
// pub(crate) fn distance_to_bounds_simd_u16_4d_squared_euclidean(
//     p1: &PT,
//     min_bound: &PT,
//     max_bound: &PT,
// ) -> A {
//     unsafe {
//         let pt_mm = _mm_loadu_ps(p1.as_ptr());
//         let max_mm = _mm_loadu_ps(max_bound.as_ptr());
//
//         let mut clamped = _mm_min_ps(max_mm, pt_mm);
//         let min_mm = _mm_load_ps(min_bound.as_ptr());
//         clamped = _mm_max_ps(min_mm, clamped);
//
//         let diff = _mm_sub_ps(pt_mm, clamped);
//
//         let squared = _mm_mul_ps(diff, diff);
//
//         let mut shuf = _mm_movehdup_ps(squared); // broadcast elements 3,1 to 2,0
//         let mut sums = _mm_add_ps(squared, shuf);
//         shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
//         sums = _mm_add_ss(sums, shuf);
//
//         f32::from_bits(_mm_extract_ps::<0>(sums) as u32)
//     }
// }


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
// pub(crate) fn extend<A: Axis, const K: usize>(min_bound: &mut [A; K], max_bound: &mut [A; K], point: &[A; K]) {
//     point.iter().zip(min_bound.iter_mut()).for_each(|(&point_coord, bound)| {
//         if point_coord < *bound {
//             *bound = point_coord;
//         }
//     });
//     point.iter().zip(max_bound.iter_mut()).for_each(|(&point_coord, bound)| {
//         if point_coord > *bound {
//             *bound = point_coord;
//         }
//     });
// }

//// SLOWER
// #[inline(never)]
// pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &PT) {
//     point.iter().zip(bounds.iter_mut()).for_each(|(&point_coord, bound)| {
//         bound.0 = bound.0.min(point_coord);
//         bound.1 = bound.1.max(point_coord);
//     });
// }

//// DOESN'T WORK
// #[inline(never)]
// pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &PT) {
//     bounds = point.iter().zip(bounds.iter()).map(|(&point_coord, (&lo, &hi))| {
//         (lo.min(point_coord), hi.max(point_coord))
//     }).collect();
// }
