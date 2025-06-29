#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![doc(html_root_url = "https://docs.rs/kiddo/5.2.0")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]

//! # Kiddo
//!
//! A high-performance, flexible, ergonomic [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) library.
//!
//! Possibly the fastest k-d tree library in the world? [See for yourself](https://sdd.github.io/kd-tree-comparison-webapp/).
//!
//! Kiddo provides:
//! - A standard floating-point k-d tree, exposed as [`kiddo::KdTree`](`crate::KdTree`), for when you may need to add or remove
//!   points to the tree after the initial construction / deserialization
//! - An [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) with performance and space advantages over the standard
//!   k-d tree, for situations where the tree does not need to be modified after creation
//! - **integer / fixed point support** via the [`fixed`](https://docs.rs/fixed/latest/fixed/) crate;
//! - **`f16` support** via the [`half`](https://docs.rs/half/latest/half/) crate;
//! - **instant zero-copy deserialization** and serialisation via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).

//!
//! Kiddo is ideal for superfast spatial / geospatial lookups and nearest-neighbour / KNN
//! queries for low-ish numbers of dimensions, where you want to ask questions such as:
//!  - Find the [nearest_n](`float::kdtree::KdTree::nearest_n`) item(s) to a query point, ordered by distance;
//!  - Find all items [within](`float::kdtree::KdTree::within`) a specified radius of a query point;
//!  - Find the ["best" n item(s) within](`float::kdtree::KdTree::best_n_within`) a specified distance of a query point, for some definition of "best",
//!    For example, "give me the 5 largest settlements within 50km of a given point, ordered by descending population", or "the 5 brightest stars
//!    within a degree of a point on the sky, ordered by brightest first".
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "5.2.0"
//! ```
//!
//! ## Usage
//! ```rust
//! use kiddo::KdTree;
//! use kiddo::SquaredEuclidean;
//! use kiddo::NearestNeighbour;
//!
//! let entries = vec![
//!     [0f64, 0f64],
//!     [1f64, 1f64],
//!     [2f64, 2f64],
//!     [3f64, 3f64]
//! ];
//!
//! // use the kiddo::KdTree type to get up and running quickly with default settings
//! let mut kdtree: KdTree<_, 2> = (&entries).into();
//!
//! // How many items are in tree?
//! assert_eq!(kdtree.size(), 4);
//!
//! // find the nearest item to [0f64, 0f64].
//! // returns a tuple of (dist, index)
//! assert_eq!(
//!     kdtree.nearest_one::<SquaredEuclidean>(&[0f64, 0f64]),
//!     NearestNeighbour { distance: 0f64, item: 0 }
//! );
//!
//! // find the nearest 3 items to [0f64, 0f64], and collect into a `Vec`
//! assert_eq!(
//!     kdtree.nearest_n::<SquaredEuclidean>(&[0f64, 0f64], 3),
//!     vec![NearestNeighbour { distance: 0f64, item: 0 }, NearestNeighbour { distance: 2f64, item: 1 }, NearestNeighbour { distance: 8f64, item: 2 }]
//! );
//! ```
//!
//! See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more in-depth examples.
//! ## Optional Features

//! The Kiddo crate exposes the following features. Any labelled as **(NIGHTLY)** are not available on `stable` Rust as they require some unstable features. You'll need to build with `nightly` in order to user them.
//! * **serde** - serialization / deserialization via [`Serde`](https://docs.rs/serde/latest/serde/)
//! * **rkyv** - zero-copy serialization / deserialization via [`Rkyv`](https://docs.rs/rkyv/0.7.45/rkyv/index.html) version 0.7.x
//! * **rkyv_08** - zero-copy serialization / deserialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) version 0.8.x
//! * `simd` **(NIGHTLY)** - enables some handwritten SIMD and pre-fetch intrinsics code within [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) that may improve performance (currently only on nearest_one with `f64`)
//! * `fixed` - enables usage of `kiddo::fixed::KdTree` for use with the `fixed` library's fixed-point number types
//!
//! **NOTE**: Support for rkyv 0.7 is now deprecated and will be removed in Kiddo v6.

#[macro_use]
extern crate doc_comment;
extern crate core;

#[doc(hidden)]
pub mod best_neighbour;
#[doc(hidden)]
pub(crate) mod common;
#[cfg(feature = "serde")]
#[doc(hidden)]
mod custom_serde;
#[cfg(feature = "fixed")]
pub mod fixed;
pub mod float;
pub mod immutable;
mod mirror_select_nth_unstable_by;
#[doc(hidden)]
pub mod nearest_neighbour;
#[doc(hidden)]
#[cfg(feature = "test_utils")]
pub mod test_utils;
pub mod traits;

mod iter;

#[doc(hidden)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod within_unsorted_iter;

#[doc(hidden)]
pub mod float_leaf_slice;
mod modified_van_emde_boas;
mod rkyv_utils;

/// A floating-point k-d tree with default parameters.
///
/// `A` is the floating point type (`f32` or `f64`, or `f16` in conjunction with the [`half`](https://docs.rs/half/latest/half/) crate).
/// `K` is the number of dimensions. See [`KdTree`](`float::kdtree::KdTree`) for details of how to use.
///
/// To manually specify more advanced parameters, use [`KdTree`](`float::kdtree::KdTree`) directly.
/// To store positions using integer or fixed-point types, use [`fixed::kdtree::KdTree`].
pub type KdTree<A, const K: usize> = float::kdtree::KdTree<A, u64, K, 32, u32>;

/// An immutable floating-point k-d tree with default parameters.
///
/// `A` is the floating point type (`f32` or `f64`, or `f16` in conjunction with the [`half`](https://docs.rs/half/latest/half/) crate).
/// `K` is the number of dimensions. See [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) for details of how to use.
///
/// To manually specify more advanced parameters, use [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) directly.
/// To store positions using integer or fixed-point types, use [`fixed::kdtree::KdTree`].
pub type ImmutableKdTree<A, const K: usize> =
    immutable::float::kdtree::ImmutableKdTree<A, u64, K, 32>;

pub use best_neighbour::BestNeighbour;
pub use float::distance::Manhattan;
pub use float::distance::SquaredEuclidean;
pub use nearest_neighbour::NearestNeighbour;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use within_unsorted_iter::WithinUnsortedIter;
