//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. [`f64`] or [`f32`] are the types that are supported for use as co-ordinates.

#[doc(hidden)]
pub mod construction;
pub mod distance;
pub mod kdtree;
#[doc(hidden)]
pub mod query;
pub(crate) mod result_collection;
