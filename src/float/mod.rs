//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. [`f64`] or [`f32`] are the types that are supported for use as co-ordinates,
//! or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled

#[doc(hidden)]
pub mod construction;
pub mod distance;
pub mod kdtree;
#[doc(hidden)]
pub mod query;
pub(crate) mod result_collection;
