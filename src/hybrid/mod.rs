//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. [`f64`] or [`f32`] are supported currently.

#[doc(hidden)]
pub mod construction;
pub mod distance;
pub mod kdtree;
pub mod neighbour;
#[doc(hidden)]
pub mod query;
