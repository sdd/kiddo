//! Fixed point kd-tree, for use when the co-ordinates of the points being stored in the tree
//! are fixed point or integers. [`u8`], [`u16`], [`u32`], and [`u64`] based fixed-point / integers are supported
//! via the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate, eg [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html) for a 16-bit fixed point number with 14 bits after the
//! decimal point.

#[doc(hidden)]
pub mod construction;
pub mod distance;
mod heap_element;
pub mod kdtree;
#[doc(hidden)]
pub mod query;
mod util;
