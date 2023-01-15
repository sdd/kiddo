use num_traits::{One, PrimInt, Unsigned, Zero};
use std::fmt::Debug;
use az::Cast;

pub trait Content: Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug + std::ops::SubAssign {}
impl<T: Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug + std::ops::SubAssign> Content for T {}

pub trait Index: PrimInt + Unsigned + Zero + Cast<usize> {
    type T: Cast<usize>;
    fn max() -> Self;
    fn min() -> Self;
    fn leaf_offset() -> Self;
    fn ilog2(self) -> Self;
    fn div_ceil(self, b: Self::T) -> Self;
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
    fn ilog2(self) -> u32 { u32::ilog2(self) }
    fn div_ceil(self, b: u32) -> u32 { u32::div_ceil(self, b) }
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
    fn ilog2(self) -> u16 { u16::ilog2(self) as u16 }
    fn div_ceil(self, b: u16) -> u16 { u16::div_ceil(self, b) }
}
