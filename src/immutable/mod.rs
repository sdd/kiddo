//! Immutable k-d trees (faster and smaller, but slower to build).
//!
//! Intended for use when the smallest possible on-disk serialized
//! size of a tree is of paramount importance, and / or the
//! fastest possible query speed is required.
//!
//! Expect improvements in query time of 10-15%, and a reduction
//! in the size of serialized trees by 33% or so on average.  
//!
//! These capabilities come with a few trade-offs:
//! 1) This tree does not provide the capability
//!    to modify its contents after it has been constructed.
//!    The co-ordinates of the points to be stored must
//!    have all been generated in advance.
//! 2) Construction time can be quite a bit slower. Typically
//!    this can be twice as long as the default `kiddo::float::kdtree::KdTree`.
//! 3) The more common that duplicate values are amongst your source points,
//!    the slower it will take to construct the tree. If you're using `f64`
//!    data that is fairly random-ish, you will probably not encounter any issues.
//!    I've successfully created 250 million node `ImmutableTree` instances with random
//!    [`f64`] data with no issues, limited only by RAM during construction.
//!    Likewise for [`f32`] based trees, up to a few million nodes.
//!
//! As per the other Kiddo float-type trees, points being stored
//! in the tree must be floats ([`f64`] or [`f32`] are supported currently).
#[doc(hidden)]
pub(crate) mod common;
pub mod float;
