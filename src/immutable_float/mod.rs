//! Immutable floating point k-d tree.
//! Intended for use when the co-ordinates of the points being stored in the tree
//! are floats ([`f64`] or [`f32`] are supported currently),
//! the contents have already been generated in advance,
//! and items will not need to be added or removed from the
//! tree once it has been constructed.
//! These constraints permit improvements to be made in both
//! space and time performance.

#[doc(hidden)]
pub mod construction;
pub mod distance;
pub mod kdtree;
pub mod neighbour;

#[doc(hidden)]
pub mod query;
