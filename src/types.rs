//! Definitions for some types that are common between the [`fixed`](crate::fixed) and [`float`](crate::float) modules
use az::Cast;
use divrem::DivCeil;
use num_traits::{One, PrimInt, Unsigned, Zero};
use std::fmt::Debug;

/// Content trait.
///
/// Must be implemented by any type that you want to use to represent the content
/// stored in a KdTree. Generally this will be `usize`, `u32`, or for trees with less
/// than 65535 points, you could use a `u16`. All these types implement `Content` with no
/// extra changes. Start off with a `usize` as that's easiest
/// since you won't need to cast to / from usize when using query results to index into
/// a Vec, and try switching tqo a smaller type and benchmarking to see if you get better
/// performance.
pub trait Content:
    Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug + std::ops::SubAssign + Sync
{
}
impl<
        T: Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug + std::ops::SubAssign + Sync,
    > Content for T
{
}

/// Implemented on u16 and u32 so that they can be used internally to index the
/// `Vec`s of Stem and Leaf nodes.
///
/// Allows `u32` or `u16` to be used as the 5th generic parameter of `float::KdTree`
/// and `fixed::KdTree`. If you will be storing fewer than `BUCKET_SIZE` * ~32k items
/// in the tree, selecting `u16` will slightly reduce the size of the Stem Nodes,
/// ensuring that more of them can be kept in the CPU cache, which may improve
/// performance (this may be offset on some architectures if it results in a
/// misalignment penalty).
pub trait Index: PrimInt + Unsigned + Zero + Cast<usize> + Sync {
    #[doc(hidden)]
    type T: Cast<usize>;
    #[doc(hidden)]
    fn max() -> Self;
    #[doc(hidden)]
    fn min() -> Self;
    #[doc(hidden)]
    fn leaf_offset() -> Self;
    #[doc(hidden)]
    fn ilog2(self) -> Self;
    #[doc(hidden)]
    fn div_ceil(self, b: Self::T) -> Self;
    #[doc(hidden)]
    fn capacity_with_bucket_size(bucket_size: usize) -> usize;
}

impl Index for u32 {
    type T = u32;
    fn max() -> u32 {
        u32::MAX
    }
    fn min() -> u32 {
        0u32
    }
    fn leaf_offset() -> u32 {
        u32::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u32 {
        u32::ilog2(self)
    }
    fn div_ceil(self, b: u32) -> u32 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        (u32::MAX - u32::MAX.overflowing_shr(1).0) as usize * bucket_size
    }
}

impl Index for u16 {
    type T = u16;
    fn max() -> u16 {
        u16::MAX
    }
    fn min() -> u16 {
        0u16
    }
    fn leaf_offset() -> u16 {
        u16::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u16 {
        u16::ilog2(self) as u16
    }
    fn div_ceil(self, b: u16) -> u16 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        (u16::MAX - u16::MAX.overflowing_shr(1).0) as usize * bucket_size
    }
}
