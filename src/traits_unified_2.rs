use fixed::traits::{Fixed, LossyFrom, LossyInto};
use fixed::types::extra::{U0, U16};
use fixed::FixedI32;

pub trait DistanceMetricUnified<A, const K: usize> {
    type Output;

    fn dist(a: &[A; K], b: &[A; K]) -> Self::Output;
    fn dist1(a: A, b: A) -> Self::Output;
}

pub struct SquaredEuclidean;
pub struct SquaredEuclideanWiden<R>(core::marker::PhantomData<R>);

#[macro_export]
macro_rules! impl_squared_euclidean_float {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for SquaredEuclidean {
            type Output = $t;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let d = ai - bi;
                        d * d
                    })
                    .fold(0.0, |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let d = a - b;
                d * d
            }
        }
    };
}

#[macro_export]
macro_rules! impl_squared_euclidean_fixed {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for SquaredEuclidean {
            type Output = $t;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let d = if ai >= bi { ai - bi } else { bi - ai };
                        d * d
                    })
                    .fold(<$t>::from_num(0), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let d = if a >= b { a - b } else { b - a };
                d * d
            }
        }
    };
}

#[macro_export]
macro_rules! impl_squared_euclidean_fixed_widening {
    ($t:ty) => {
        impl<R, const K: usize> DistanceMetricUnified<$t, K> for SquaredEuclideanWiden<R>
        where
            R: Fixed + LossyFrom<$t>,
        {
            type Output = R;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let ai: Self::Output = ai.lossy_into();
                        let bi: Self::Output = bi.lossy_into();

                        let d = if ai >= bi { ai - bi } else { bi - ai };
                        d * d
                    })
                    .fold(<Self::Output>::from_num(0), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let a: Self::Output = a.lossy_into();
                let b: Self::Output = b.lossy_into();

                let d: Self::Output = if a >= b { a - b } else { b - a };
                d * d
            }
        }
    };
}

impl_squared_euclidean_float!(f32);
impl_squared_euclidean_float!(f64);
impl_squared_euclidean_fixed!(FixedI32<U16>);
impl_squared_euclidean_fixed!(FixedI32<U0>);

impl_squared_euclidean_fixed_widening!(FixedI32<U16>);
impl_squared_euclidean_fixed_widening!(FixedI32<U0>);

#[inline]
pub fn calc_dists(content_points: &[[f32; 64]; 3], acc: &mut [f32; 64], query: &[f32; 3]) {
    // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
    (0..3).for_each(|dim| {
        (0..64).for_each(|idx| {
            acc[idx] +=
                (content_points[dim][idx] - query[dim]) * (content_points[dim][idx] - query[dim]);
        });
    });
}

pub(crate) fn update_nearest(
    dists: &[f32; 64],
    items: &[usize; 64],
    best_dist: &mut f32,
    best_item: &mut usize,
) {
    // Autovectorizes with 256bit vectors on x86_64 where available
    // 341 loops (1 item per loop, unrolled x 3) of 4-8 instructions per item
    let (leaf_best_item, leaf_best_dist) = dists
        .iter()
        .enumerate()
        .min_by(|&(_, &a), (_, b)| a.partial_cmp(*b).unwrap())
        .unwrap();

    // 6 instructions, 1 branch
    if *leaf_best_dist < *best_dist {
        *best_dist = *leaf_best_dist;
        *best_item = items[leaf_best_item];
    }
}

/// Hook function for cargo-asm to verify autovectorization with concrete types.
/// This should vectorize perfectly on zen5 and serve as a baseline for the generic trait implementation.
#[inline(never)]
pub fn bench_update_nearest_f32_64(
    content_points: &[[f32; 64]; 3],
    items: &[usize; 64],
    query: &[f32; 3],
) -> (f32, usize) {
    let mut best_dist = f32::INFINITY;
    let mut best_item = 0usize;

    let mut acc = [0f32; 64];

    calc_dists(content_points, &mut acc, query);

    update_nearest(&acc, items, &mut best_dist, &mut best_item);

    (best_dist, best_item)
}
