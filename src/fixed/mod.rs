//! Fixed point KD Tree, for use when the co-ordinates of the points being stored in the tree
//! are fixed point or integers. `u8`, `u16`, `u32`, and `u64` based fixed-point / integers are supported
//! via the Fixed crate, eg `FixedU16<U14>` for a 16-bit fixed point number with 14 bits after the
//! decimal point.

#[doc(hidden)]
pub mod construction;
pub mod distance;
mod heap_element;
pub mod kdtree;
#[doc(hidden)]
pub mod query;
mod util;
