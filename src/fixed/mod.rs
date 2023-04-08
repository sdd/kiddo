//! Fixed point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are fixed point or integers. [`u8`], [`u16`], [`u32`], and [`u64`] based fixed-point / integers are supported
//! via the Fixed crate, eg [`FixedU16<U14>`](fixed::FixedU16<U14>) for a 16-bit fixed point number with 14 bits after the
//! decimal point.

#[doc(hidden)]
pub mod construction;
pub mod distance;
pub mod kdtree;
pub mod neighbour;
#[doc(hidden)]
pub mod query;
