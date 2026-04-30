#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_prefetch))]
#![cfg_attr(feature = "simd", feature(target_feature_inline_always))]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![doc(html_root_url = "https://docs.rs/kiddo/6.0.0-alpha.1")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]
#![allow(clippy::pointers_in_nomem_asm_block)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

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
//!  - Find the [nearest_n](`mutable::float::kdtree::KdTree`::nearest_n`) item(s) to a query point, ordered by distance;
//!  - Find all items [within](`mutable::float::kdtree::KdTree`::within`) a specified radius of a query point;
//!  - Find the ["best" n item(s) within](`mutable::float::kdtree::KdTree`::best_n_within`) a specified distance of a query point, for some definition of "best",
//!    For example, "give me the 5 largest settlements within 50km of a given point, ordered by descending population", or "the 5 brightest stars
//!    within a degree of a point on the sky, ordered by brightest first".
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "5.3.0"
//! ```
//!
//! ## Usage
//! ```rust
//! use std::num::NonZero;
//!
//! use kiddo::leaf_strategy::VecOfArrays;
//! use kiddo::SquaredEuclidean;
//! use kiddo::NearestNeighbour;
//! use kiddo::{Eytzinger, KdTree};
//!
//! let entries = vec![
//!     [0f64, 0f64],
//!     [1f64, 1f64],
//!     [2f64, 2f64],
//!     [3f64, 3f64]
//! ];
//!
//! type Tree = KdTree<f64, usize, Eytzinger<2>, VecOfArrays<f64, usize, 2, 32>, 2, 32>;
//! let kdtree: Tree = KdTree::new_from_slice(&entries);
//!
//! // How many items are in tree?
//! assert_eq!(kdtree.size(), 4);
//!
//! // find the nearest item to [0f64, 0f64].
//! // returns a tuple of (dist, index)
//! assert_eq!(
//!     kdtree.nearest_one::<SquaredEuclidean<f64>>(&[0f64, 0f64]),
//!     (0f64, 0)
//! );
//!
//! // find the nearest 3 items to [0f64, 0f64], and collect into a `Vec`
//! assert_eq!(
//!     kdtree.nearest_n::<SquaredEuclidean<f64>>(
//!         &[0f64, 0f64],
//!         NonZero::new(3usize).unwrap(),
//!         true
//!     ),
//!     vec![NearestNeighbour { distance: 0f64, item: 0 }, NearestNeighbour { distance: 2f64, item: 1 }, NearestNeighbour { distance: 8f64, item: 2 }]
//! );
//! ```
//!
//! See the [examples documentation](https://github.com/sdd/kiddo/tree/master/examples) for some more in-depth examples.
//! ## Optional Features

//! The Kiddo crate exposes the following features. Any labelled as **(NIGHTLY)** are not available on `stable` Rust as they require some unstable features. You'll need to build with `nightly` in order to user them.
//! * **serde** - serialization / deserialization via [`Serde`](https://docs.rs/serde/latest/serde/)
//! * **rkyv_08** - zero-copy serialization / deserialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) version 0.8.x
//! * `simd` **(NIGHTLY)** - enables some handwritten SIMD and pre-fetch intrinsics code within [`ImmutableKdTree`](`immutable::float::kdtree::ImmutableKdTree`) that may improve performance (currently only on nearest_one with `f64`)
//! * `fixed` - enables usage of `kiddo::mutable::fixed::KdTree` for use with the `fixed` library's fixed-point number types
//!
//! **NOTE**: Support for rkyv 0.7 was removed in Kiddo v6.

extern crate core;
extern crate doc_comment;

// #[doc(hidden)]
// #[cfg(feature = "serde")]
// #[doc(hidden)]
// mod custom_serde;

pub mod huge_pages;

mod mirror_select_nth_unstable_by;

#[cfg(feature = "rkyv_08")]
mod rkyv;

#[doc(hidden)]
#[cfg(feature = "test_utils")]
pub mod test_utils;
pub mod traits;

/// Leaf storage strategies for the kd-tree
pub mod leaf_strategy;

/// Leaf view abstraction for accessing leaf data
pub mod leaf_view;

/// Chunked Leaf view abstraction for accessing leaf data
pub(crate) mod leaf_view_chunked;

/// Stem Orderings
pub mod stem_strategy;
pub use traits::StemStrategy;

/// Distance metrics
pub mod dist;
pub mod kd_tree;

/// Structs that are returned as query results
pub mod results;

pub mod traits_unified_2;

pub use crate::dist::{DotProduct, Manhattan, SquaredEuclidean};
pub use crate::kd_tree::KdTree;
pub use crate::kd_tree::WithinUnsortedIter;
pub use results::best_neighbour::BestNeighbour;
pub use results::nearest_neighbour::NearestNeighbour;

pub use crate::stem_strategy::Eytzinger;
