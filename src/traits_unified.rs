use crate::traits_unified::sealed::{FixedMarker, FloatMarker};
use aligned_vec::AVec;
use std::fmt::Debug;

pub trait Basics: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Basics for T where T: Copy + Debug + Default + Send + Sync + 'static {}

mod sealed {
    use std::iter::Sum;
    use std::marker::PhantomData;
    use std::ops::AddAssign;

    use fixed::prelude::*;
    use fixed::traits::Fixed;
    use fixed::{FixedI16, FixedI32, FixedI64, FixedI8, FixedU16, FixedU32, FixedU64, FixedU8};
    use num_traits::float::FloatCore;

    use crate::traits_unified::Basics;

    pub trait Sealed {}
    pub(crate) trait IsFloat: Sealed + Basics + FloatCore + AddAssign + Sum {}
    pub(crate) trait IsFixed: Sealed + Basics + Fixed + ToFixed {}

    pub struct FloatMarker<A: IsFloat + Basics>(PhantomData<A>);
    pub struct FixedMarker<A: IsFixed + Basics>(PhantomData<A>);

    impl Sealed for f32 {}
    impl IsFloat for f32 {}
    impl Sealed for f64 {}
    impl IsFloat for f64 {}

    impl<T> Sealed for FixedI8<T> {}
    impl<T: Basics> IsFixed for FixedI8<T> where FixedI8<T>: Fixed {}
    impl<T> Sealed for FixedU8<T> {}
    impl<T: Basics> IsFixed for FixedU8<T> where FixedU8<T>: Fixed {}
    impl<T> Sealed for FixedI16<T> {}
    impl<T: Basics> IsFixed for FixedI16<T> where FixedI16<T>: Fixed {}
    impl<T> Sealed for FixedU16<T> {}
    impl<T: Basics> IsFixed for FixedU16<T> where FixedU16<T>: Fixed {}
    impl<T> Sealed for FixedI32<T> {}
    impl<T: Basics> IsFixed for FixedI32<T> where FixedI32<T>: Fixed {}
    impl<T> Sealed for FixedU32<T> {}
    impl<T: Basics> IsFixed for FixedU32<T> where FixedU32<T>: Fixed {}
    impl<T> Sealed for FixedI64<T> {}
    impl<T: Basics> IsFixed for FixedI64<T> where FixedI64<T>: Fixed {}
    impl<T> Sealed for FixedU64<T> {}
    impl<T: Basics> IsFixed for FixedU64<T> where FixedU64<T>: Fixed {}
}

/// AxisUnified trait encapsulates the features that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on Kiddo's trees. It is used to represent the co-ordinates of
/// points stored in the tree and queries made against the tree.
///
/// Implementors:
/// * standard floats, [`f64`] and [`f32`]
/// * with the `half` feature enabled, [`f16`](https://docs.rs/half/latest/half/struct.f16.html)
///   in conjunction with the [`half`](https://docs.rs/half/latest/half) crate
/// * with the `fixed` feature enabled, [`Fixed`] in conjunction with the
///   [`fixed`](https://docs.rs/half/latest/fixed) crate
pub trait AxisUnified {
    type NumType: Copy;

    /// Return zero for the coordinate type.
    fn zero() -> Self::NumType;

    /// Absolute/saturating difference along one axis.
    fn saturating_dist(a: Self::NumType, other: Self::NumType) -> Self::NumType;
}

impl<T> AxisUnified for FloatMarker<T>
where
    T: sealed::IsFloat,
{
    type NumType = T;

    #[inline(always)]
    fn zero() -> T {
        T::zero()
    }

    #[inline(always)]
    fn saturating_dist(a: T, b: T) -> T {
        a - b
    }
}

impl<T> AxisUnified for FixedMarker<T>
where
    T: sealed::IsFixed,
{
    type NumType = T;

    #[inline(always)]
    fn zero() -> Self::NumType {
        T::ZERO
    }

    #[inline(always)]
    fn saturating_dist(a: Self::NumType, b: Self::NumType) -> Self::NumType {
        if a >= b {
            a - b
        } else {
            b - a
        }
    }
}

/// Unified distance metric:
/// - A is the coordinate type (implements AxisUnified).
/// - K is the dimension.
/// - R is the distance/accumulator type, chosen by the caller (or via aliases).
pub trait DistanceUnified<A, const K: usize, R>
where
    A: AxisUnified,
    R: Basics + PartialOrd,
{
    /// Full K-d distance between two points.
    fn dist(a: &[A::NumType; K], b: &[A::NumType; K]) -> R;

    /// Single-axis contribution (must be consistent with dist).
    fn dist1(a: A::NumType, b: A::NumType) -> R;

    /// Accumulator update semantics for pruning distances.
    /// For floats this is rd + delta; for fixed this should be a saturating add.
    fn rd_update(rd: R, delta: R) -> R;
}

/// Trait to describe how to widen a coordinate type to an accumulator type.
/// For floats, this is often identity (f32 -> f32, f64 -> f64).
/// For fixed-point, this might widen to a larger fixed type to avoid overflow.
pub trait Widen<A: AxisUnified> {
    type Output: Basics + PartialOrd;

    /// Widen a single coordinate value to the accumulator type.
    fn widen(val: A::NumType) -> Self::Output;
}

// Float widening is typically identity
impl<T> Widen<FloatMarker<T>> for FloatMarker<T>
where
    T: sealed::IsFloat,
{
    type Output = T;

    #[inline(always)]
    fn widen(val: T) -> T {
        val
    }
}

// Fixed widening is typically identity for now (can be extended later)
impl<T> Widen<FixedMarker<T>> for FixedMarker<T>
where
    T: sealed::IsFixed,
{
    type Output = T;

    #[inline(always)]
    fn widen(val: T) -> T {
        val
    }
}

// Specialized widening: FixedI16<8> -> FixedI32<8>
use fixed::types::{I16F16, I32F0};

impl Widen<FixedMarker<I16F16>> for FixedMarker<I32F0> {
    type Output = I32F0;

    #[inline(always)]
    fn widen(val: I16F16) -> I32F0 {
        I32F0::from_num(val)
    }
}

/// Squared Euclidean distance metric.
pub struct SquaredEuclideanUnified;

#[repr(align(512))]
struct TestBucket([[f32; 64]; 3]);

#[repr(align(512))]
struct TestAcc([f32; 64]);

/// Benchmark function to test autovectorization of bucket scan.
/// This simulates scanning 64 points in 3D space to find the nearest neighbor.
/// Use with cargo-asm to inspect the generated assembly.
#[inline(never)]
pub fn bench_bucket_scan_f32_3d_64(
    bucket: &TestBucket,
    query: &[f32; 3],
    acc: &mut TestAcc,
) -> (f32, usize) {
    type Axis = FloatMarker<f32>;

    let mut best_dist = f32::INFINITY;
    let mut best_idx = 0;

    // Compute distances dimension by dimension (enables vectorization)
    // bucket.0.iter().enumerate().for_each(|(i, bucket)| {
    //     acc.0.iter_mut().zip(bucket.iter()).for_each(|(acc, val)| {
    //         // *acc += <SquaredEuclideanUnified as DistanceUnified<Axis, 3, f32>>::dist1(*bucket, query[i]);
    //         *acc += (*val - query[i]) * (*val - query[i]);
    //     })
    // });

    #[inline(always)]
    fn accumulate_one_dim(acc: &mut [f32; 64], bucket: &[f32; 64], q: f32) {
        for i in 0..64 {
            acc[i] += (bucket[i] - q) * (bucket[i] - q);
        }
    }

    for dim in 0..3 {
        accumulate_one_dim(&mut acc.0, &bucket.0[dim], query[dim]);
    }

    // Find minimum
    // Autovectorizes with 256bit vectors on x86_64 where available
    // 341 loops (1 item per loop, unrolled x 3) of 4-8 instructions per item
    let (leaf_best_item, leaf_best_dist) = acc
        .0
        .into_iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // 6 instructions, 1 branch
    if leaf_best_dist < best_dist {
        best_dist = leaf_best_dist;
        best_idx = leaf_best_item;
    }

    (best_dist, best_idx)
}

impl<A, const K: usize> DistanceUnified<A, K, <A as Widen<A>>::Output> for SquaredEuclideanUnified
where
    A: AxisUnified + Widen<A>,
    <A as Widen<A>>::Output: Basics
        + PartialOrd
        + core::ops::Add<Output = <A as Widen<A>>::Output>
        + core::ops::Mul<Output = <A as Widen<A>>::Output>,
{
    #[inline(always)]
    fn dist(a: &[A::NumType; K], b: &[A::NumType; K]) -> <A as Widen<A>>::Output {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                let diff = A::saturating_dist(ai, bi);
                let widened = A::widen(diff);
                widened * widened
            })
            .fold(A::widen(A::zero()), |acc, x| acc + x)
    }

    #[inline(always)]
    fn dist1(a: A::NumType, b: A::NumType) -> <A as Widen<A>>::Output {
        let diff = A::saturating_dist(a, b);
        let widened = A::widen(diff);
        widened * widened
    }

    #[inline(always)]
    fn rd_update(
        rd: <A as Widen<A>>::Output,
        delta: <A as Widen<A>>::Output,
    ) -> <A as Widen<A>>::Output {
        rd + delta
    }
}

/// Strategy for how leaf storage is laid out and constructed.
/// - AX: Axis marker implementing AxisUnified (selects float or fixed semantics).
/// - T: item/content type stored alongside points.
/// - SS: stem layout strategy used to map split values into KdTree::stems (external).
/// - K: dimensionality.
/// - B: nominal bucket size (strategies may use or ignore it).
pub trait LeafStrategy<AX, T, SS, const K: usize, const B: usize>
where
    AX: AxisUnified,
    T: Basics,
    SS: crate::traits::StemStrategy,
{
    /// Coordinate scalar type.
    type Num;

    // ---- Construction ----

    /// Create a builder with an intended capacity (in points).
    fn new_builder(capacity: usize) -> Self;

    /// Bulk-build from a slice of points. Implementations should:
    /// - write split values into `stems` at indices determined by `stem_strategy`,
    /// - lay out leaf storage according to the strategy,
    /// - return the max stem level reached (for later traversal).
    fn bulk_build_from_slice(
        &mut self,
        source: &[[Self::Num; K]],
        stems: &mut AVec<Self::Num>,
        stem_strategy: SS,
    ) -> i32;

    /// Finalization hook (e.g., trim stems or compact internal buffers).
    fn finalize(
        &mut self,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
        max_stem_level: i32,
    );

    // ---- Mutation (optional; immutable strategies may panic or be no-ops) ----

    /// Add a point/item pair into the structure, updating stems as necessary.
    fn add_point(
        &mut self,
        point: &[Self::Num; K],
        item: T,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
    );

    /// Remove a specific point/item pair; returns number removed.
    fn remove_point(&mut self, point: &[Self::Num; K], item: T) -> usize;

    // ---- Introspection / minimal accessors ----

    /// Total number of stored items.
    fn size(&self) -> usize;

    /// Number of leaves maintained by the strategy (buckets/extents).
    fn leaf_count(&self) -> usize;

    /// Number of items in a given leaf.
    fn leaf_len(&self, leaf_idx: usize) -> usize;

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K>;
}

pub struct DummyLeafStrategy {}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B> for DummyLeafStrategy
where
    AX: AxisUnified,
    T: Basics,
    SS: crate::traits::StemStrategy,
{
    type Num = ();

    fn new_builder(capacity: usize) -> Self {
        unimplemented!()
    }

    fn bulk_build_from_slice(
        &mut self,
        source: &[[Self::Num; K]],
        stems: &mut AVec<Self::Num>,
        stem_strategy: SS,
    ) -> i32 {
        unimplemented!()
    }

    fn finalize(
        &mut self,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
        max_stem_level: i32,
    ) {
        unimplemented!()
    }

    fn add_point(
        &mut self,
        point: &[Self::Num; K],
        item: T,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
    ) {
        unimplemented!()
    }

    fn remove_point(&mut self, point: &[Self::Num; K], item: T) -> usize {
        unimplemented!()
    }

    fn size(&self) -> usize {
        unimplemented!()
    }

    fn leaf_count(&self) -> usize {
        unimplemented!()
    }

    fn leaf_len(&self, leaf_idx: usize) -> usize {
        unimplemented!()
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K> {
        unimplemented!()
    }
}

pub type LeafView<'a, AX, T, const K: usize> = ([&'a [AX]; K], &'a [T]);

#[cfg(test)]
mod tests {
    use super::*;
    use fixed::types::{I16F16, I32F0};

    #[test]
    fn test_squared_euclidean_f64_2d() {
        type Axis = FloatMarker<f64>;

        let a = [3.0, 4.0];
        let b = [0.0, 0.0];

        // Distance should be 3^2 + 4^2 = 9 + 16 = 25
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 2, f64>>::dist(&a, &b);
        assert_eq!(dist, 25.0);
    }

    #[test]
    fn test_squared_euclidean_f64_3d() {
        type Axis = FloatMarker<f64>;

        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 8.0];

        // Distance should be (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 3, f64>>::dist(&a, &b);
        assert_eq!(dist, 50.0);
    }

    #[test]
    fn test_squared_euclidean_f64_dist1() {
        type Axis = FloatMarker<f64>;

        let a = 5.0;
        let b = 2.0;

        // Single axis distance should be (5-2)^2 = 9
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 2, f64>>::dist1(a, b);
        assert_eq!(dist, 9.0);
    }

    #[test]
    fn test_squared_euclidean_f64_rd_update() {
        type Axis = FloatMarker<f64>;

        let rd = 10.0;
        let delta = 5.0;

        let result =
            <SquaredEuclideanUnified as DistanceUnified<Axis, 2, f64>>::rd_update(rd, delta);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_squared_euclidean_fixed_i16_2d() {
        type Axis = FixedMarker<I16F16>;

        let a = [I16F16::from_num(3), I16F16::from_num(4)];
        let b = [I16F16::from_num(0), I16F16::from_num(0)];

        // Distance should be 3^2 + 4^2 = 9 + 16 = 25
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 2, I16F16>>::dist(&a, &b);
        assert_eq!(dist, I16F16::from_num(25));
    }

    #[test]
    fn test_squared_euclidean_fixed_i16_3d() {
        type Axis = FixedMarker<I16F16>;

        let a = [
            I16F16::from_num(1),
            I16F16::from_num(2),
            I16F16::from_num(3),
        ];
        let b = [
            I16F16::from_num(4),
            I16F16::from_num(6),
            I16F16::from_num(8),
        ];

        // Distance should be (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 3, I16F16>>::dist(&a, &b);
        assert_eq!(dist, I16F16::from_num(50));
    }

    #[test]
    fn test_squared_euclidean_fixed_i16_dist1() {
        type Axis = FixedMarker<I16F16>;

        let a = I16F16::from_num(5);
        let b = I16F16::from_num(2);

        // Single axis distance should be (5-2)^2 = 9
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 2, I16F16>>::dist1(a, b);
        assert_eq!(dist, I16F16::from_num(9));
    }

    #[test]
    fn test_squared_euclidean_fixed_i16_fractional() {
        type Axis = FixedMarker<I16F16>;

        let a = [I16F16::from_num(1.5), I16F16::from_num(2.5)];
        let b = [I16F16::from_num(0.5), I16F16::from_num(0.5)];

        // Distance should be (1.5-0.5)^2 + (2.5-0.5)^2 = 1 + 4 = 5
        let dist = <SquaredEuclideanUnified as DistanceUnified<Axis, 2, I16F16>>::dist(&a, &b);
        assert_eq!(dist, I16F16::from_num(5));
    }

    #[test]
    fn test_widening_i16_to_i32_2d() {
        // Test widening from I16F16 to I32F0 for accumulation
        // Use values that would overflow I16F16 when squared
        // I16F16 max value is ~32767.99
        // Use 300 so that 300^2 = 90000 > 65536, ensuring both the squaring
        // and accumulation steps require widening to work properly
        type CoordAxis = FixedMarker<I16F16>;
        type AccumAxis = FixedMarker<I32F0>;

        let a = [I16F16::from_num(300), I16F16::from_num(250)];
        let b = [I16F16::from_num(0), I16F16::from_num(0)];

        // Manually compute using widening
        let diff0 = CoordAxis::saturating_dist(a[0], b[0]);
        let diff1 = CoordAxis::saturating_dist(a[1], b[1]);

        let widened0 = <AccumAxis as Widen<CoordAxis>>::widen(diff0);
        let widened1 = <AccumAxis as Widen<CoordAxis>>::widen(diff1);

        let squared0 = widened0 * widened0;
        let squared1 = widened1 * widened1;

        let dist = squared0 + squared1;

        // Distance should be 300^2 + 250^2 = 90000 + 62500 = 152500
        // 300^2 = 90000 alone exceeds 65536, demonstrating widening is essential
        // This would overflow I16F16 (max ~32767) but fits in I32F0
        assert_eq!(dist, I32F0::from_num(152500));
    }

    #[test]
    fn test_widening_i16_to_i32_3d() {
        // Test with even larger values that would definitely overflow without widening
        type CoordAxis = FixedMarker<I16F16>;
        type AccumAxis = FixedMarker<I32F0>;

        let a = [
            I16F16::from_num(150),
            I16F16::from_num(180),
            I16F16::from_num(200),
        ];
        let b = [
            I16F16::from_num(0),
            I16F16::from_num(0),
            I16F16::from_num(0),
        ];

        // Manually compute using widening
        let diff0 = CoordAxis::saturating_dist(a[0], b[0]);
        let diff1 = CoordAxis::saturating_dist(a[1], b[1]);
        let diff2 = CoordAxis::saturating_dist(a[2], b[2]);

        let widened0 = <AccumAxis as Widen<CoordAxis>>::widen(diff0);
        let widened1 = <AccumAxis as Widen<CoordAxis>>::widen(diff1);
        let widened2 = <AccumAxis as Widen<CoordAxis>>::widen(diff2);

        let squared0 = widened0 * widened0;
        let squared1 = widened1 * widened1;
        let squared2 = widened2 * widened2;

        let dist = squared0 + squared1 + squared2;

        // Distance should be 150^2 + 180^2 + 200^2 = 22500 + 32400 + 40000 = 94900
        // Each squared term > 22500, which would overflow I16F16 individually
        // The sum would definitely overflow I16F16
        assert_eq!(dist, I32F0::from_num(94900));
    }

    #[test]
    fn test_widening_preserves_precision() {
        type CoordAxis = FixedMarker<I16F16>;
        type AccumAxis = FixedMarker<I32F0>;

        // Test that fractional parts are preserved during widening
        let val = I16F16::from_num(3.75);
        let widened = <AccumAxis as Widen<CoordAxis>>::widen(val);

        // After widening and rounding to I32F0, we should get 4 (or 3, depending on rounding)
        // I32F0 has no fractional bits, so 3.75 will be truncated to 3
        assert_eq!(widened, I32F0::from_num(3));
    }
}
