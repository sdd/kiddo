#[cfg(feature = "simd")]
use std::arch::x86_64::{ _mm256_load_pd, _mm256_max_pd, _mm256_min_pd, _mm256_store_pd, _mm_load_ps, _mm_max_ps, _mm_min_ps, _mm_store_ps};

use crate::sok::Axis;

pub(crate) trait BoundsExtender<A: Axis, const K: usize> {
    fn extend(min_bound: &mut [A; K], max_bound: &mut [A; K], point: &[A; K]);
}

impl<A: Axis, const K: usize> BoundsExtender<A, K> for [A; K] {
    default fn extend(min_bound: &mut [A; K], max_bound: &mut [A; K], point: &[A; K]) {
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
}

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64"))]
#[allow(clippy::missing_safety_doc)]
impl<A: Axis, const K: usize> BoundsExtender<A, K> for [f32; 3] {
    #[target_feature(enable = "sse")]
    unsafe fn extend(min_bound: &mut [f32; 3], max_bound: &mut [f32; 3], point: &[f32; 3]) {
        let pt_mm = _mm_load_ps(point.as_ptr());
        let mut max_mm = _mm_load_ps(max_bound.as_ptr());
        let mut min_mm = _mm_load_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_store_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_store_ps(max_bound.as_mut_ptr(), max_mm);
    }
}

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64"))]
#[allow(clippy::missing_safety_doc)]
impl<A: Axis, const K: usize> BoundsExtender<A, K> for [f32; 4] {
    #[target_feature(enable = "sse")]
    unsafe fn extend(min_bound: &mut [f32], max_bound: &mut [f32], point: &[f32]) {
        let pt_mm = _mm_load_ps(point.as_ptr());
        let mut max_mm = _mm_load_ps(max_bound.as_ptr());
        let mut min_mm = _mm_load_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_store_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_store_ps(max_bound.as_mut_ptr(), max_mm);
    }
}

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64"))]
#[allow(clippy::missing_safety_doc)]
impl<A: Axis, const K: usize> BoundsExtender<A, K> for [f64; 3] {
    #[target_feature(enable = "avx")]
    unsafe fn extend(min_bound: &mut [f64], max_bound: &mut [f64], point: &[f64]) {
        let pt_mm = _mm256_load_pd(point.as_ptr());
        let mut max_mm = _mm256_load_pd(max_bound.as_ptr());
        let mut min_mm = _mm256_load_pd(min_bound.as_ptr());

        min_mm = _mm256_min_pd(min_mm, pt_mm);
        max_mm = _mm256_max_pd(max_mm, pt_mm);

        _mm256_store_pd(min_bound.as_mut_ptr(), min_mm);
        _mm256_store_pd(max_bound.as_mut_ptr(), max_mm);
    }
}

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64"))]
#[allow(clippy::missing_safety_doc)]
impl<A: Axis, const K: usize> BoundsExtender<A, K> for [f64; 4] {
    #[target_feature(enable = "avx")]
    unsafe fn extend(min_bound: &mut [f64], max_bound: &mut [f64], point: &[f64]) {
        let pt_mm = _mm256_load_pd(point.as_ptr());
        let mut max_mm = _mm256_load_pd(max_bound.as_ptr());
        let mut min_mm = _mm256_load_pd(min_bound.as_ptr());

        min_mm = _mm256_min_pd(min_mm, pt_mm);
        max_mm = _mm256_max_pd(max_mm, pt_mm);

        _mm256_store_pd(min_bound.as_mut_ptr(), min_mm);
        _mm256_store_pd(max_bound.as_mut_ptr(), max_mm);
    }
}

/*// SLOWER
#[inline(never)]
pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
    point.iter().zip(bounds.iter_mut()).for_each(|(&point_coord, bound)| {
        if point_coord < bound.0 {
            bound.0 = point_coord;
        }
        if point_coord > bound.1 {
            bound.1 = point_coord;
        }
    });
}

// SLOWER
#[inline(never)]
pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
    point.iter().zip(bounds.iter_mut()).for_each(|(&point_coord, bound)| {
        bound.0 = bound.0.min(point_coord);
        bound.1 = bound.1.max(point_coord);
    });
}

// DOESN'T WORK
#[inline(never)]
pub(crate) fn extend<A: Axis, const K: usize>(bounds: &mut [(A, A); K], point: &[A; K]) {
    bounds = point.iter().zip(bounds.iter()).map(|(&point_coord, (&lo, &hi))| {
        (lo.min(point_coord), hi.max(point_coord))
    }).collect();
}*/

#[cfg(test)]
mod tests {
    use crate::bounds_extender::BoundsExtender;

    #[test]
    fn bounds_checker_works_2d_f32() {
        let mut min: [f32; 2] = [1f32, 100f32];
        let mut max: [f32; 2] = [10f32, 200f32];
        let point: [f32; 2] = [0f32, 300f32];

        BoundsExtender::<f32, 2usize>::extend::<>(&mut min, &mut max, &point);

        assert_eq!(min[0], 0f32);
        assert_eq!(min[1], 10f32);
        assert_eq!(max[0], 100f32);
        assert_eq!(max[1], 300f32);
    }

}
