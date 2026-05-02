use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Sub};

use fixed::types::extra::{U0, U16, U8};
use fixed::{FixedI32, FixedU16};

use num_traits::Float;

/// Trait for coordinate/axis types used in the [`KdTree`] (the `AX` type parameter)
///
/// This trait must be implemented on a type for it to be usable as the coordinates
/// for points stored in a [`KdTree`].
///
/// * By default, it is defined for `f32` and `f64`.
/// * Enabling the `f16` feature adds support for the `f16` type from the `half` crate.
/// * Enabling the `fixed` feature adds support for the following types from the `fixed` crate:
///     - `FixedI32<U16>`
///     - `FixedI32<U0>`
///     - `FixedU16<U8>`
///     - `FixedU16<U0>`
///     - `FixedU8<U0>`
///
/// (If you have a requirement to support a `fixed` type that is not listed, please open an issue on GitHub.)
pub trait Axis:
    Copy
    + PartialEq
    + PartialOrd
    + Sub<Output = Self>
    + AddAssign<Self>
    + Debug
    + Display
    + crate::stem_strategy::CompareBlock3
    + crate::stem_strategy::CompareBlock4
{
    /// Coordinate scalar type stored in the tree and queries.
    type Coord: Copy;

    /// Zero coord.
    fn zero() -> Self::Coord;

    /// Maximum coord value.
    fn max_value() -> Self::Coord;

    /// Minimum coord value.
    fn min_value() -> Self::Coord;

    /// If coord is max value or not.
    fn is_max_value(coord: Self::Coord) -> bool;

    /// Compares two coordinate values.
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering;

    /// Absolute/saturating difference along one axis, in coord units.
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Saturating addition of two coordinate values.
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Returns the maximum of two coordinate values.
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord;
}

/// Macro to implement AxisUnified for floating-point types.
#[macro_export]
macro_rules! impl_axis_float {
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_float!($t);

        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    ($t:ty) => {
        impl Axis for $t {
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                0.0
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <$t>::infinity()
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                <$t>::neg_infinity()
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord.is_infinite() && coord.is_sign_positive()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                // debug_assert!(
                //     a.is_finite() && b.is_finite(),
                //     "NaNs / Infinities should not be present in axis coordinates"
                // );
                if a < b {
                    std::cmp::Ordering::Less
                } else if a > b {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                (a - b).abs()
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a + b
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

/// Macro to implement AxisUnified for fixed-point types.
macro_rules! impl_axis_fixed {
    // Pattern with SIMD block support
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_fixed!($t); // First implement the basic AxisUnified trait

        // Then implement SIMD block support for each specified block size
        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    // Base pattern without SIMD block support (uses default unimplemented!() from traits)
    ($t:ty) => {
        impl Axis for $t {
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                <$t>::from_num(0)
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <Self::Coord>::MAX
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                unimplemented!("min_value not yet implemented for fixed point types")
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord == <$t>::max_value()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                a.cmp(&b)
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                if a >= b {
                    a - b
                } else {
                    b - a
                }
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.saturating_add(b)
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

// Axis impls stay as they are.
impl_axis_float!(f32);
impl_axis_float!(f64);

#[cfg(feature = "f16")]
impl Axis for half::f16 {
    type Coord = half::f16;

    #[inline(always)]
    fn zero() -> Self::Coord {
        half::f16::from_f32(0.0)
    }

    #[inline(always)]
    fn max_value() -> Self::Coord {
        half::f16::from_f32(f32::INFINITY)
    }

    #[inline(always)]
    fn min_value() -> Self::Coord {
        half::f16::from_f32(f32::NEG_INFINITY)
    }

    #[inline(always)]
    fn is_max_value(coord: Self::Coord) -> bool {
        coord.is_infinite() && coord.is_sign_positive()
    }

    #[allow(clippy::ifs_same_cond)]
    #[inline(always)]
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
        if a < b {
            std::cmp::Ordering::Less
        } else if b > a {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    #[inline(always)]
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        (a - b).abs()
    }

    #[inline(always)]
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        a + b
    }

    #[inline(always)]
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        if a > b {
            a
        } else {
            b
        }
    }
}
impl_axis_fixed!(FixedI32<U16>);
impl_axis_fixed!(FixedI32<U0>);
impl_axis_fixed!(FixedU16<U8>);
