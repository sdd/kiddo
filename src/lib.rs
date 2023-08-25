//#![cfg_attr(feature = "simd", feature(stdsimd))]
#![cfg_attr(feature = "immutable", feature(allocator_api))]
#![cfg_attr(feature = "immutable", feature(int_roundings))]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![doc(html_root_url = "https://docs.rs/kiddo/3.0.0-beta.1")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]

//! # Kiddo
//!
//! A high-performance, flexible, ergonomic [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) library.
//!
//! Possibly the fastest k-d tree library in the world? [See for yourself](https://sdd.github.io/kd-tree-comparison-webapp/).
//!
//! Version 2 and onwards is a complete rewrite over the previous v0.6.x codebase, providing:
//! - a new internal architecture for **much-improved performance**;
//! - Added **integer / fixed point support** via the [`Fixed`](https://docs.rs/fixed/latest/fixed/) library;
//! - **instant zero-copy deserialization** and serialization via [`Rkyv`](https://docs.rs/rkyv/latest/rkyv/) ([`Serde`](https://docs.rs/serde/latest/serde/) still available).
//! - See the [changelog](https://github.com/sdd/kiddo/blob/master/CHANGELOG.md) for a detailed run-down on all the changes made in v2.
//!
//! Kiddo is ideal for super-fast spatial / geospatial lookups and nearest-neighbour / KNN
//! queries for low-ish numbers of dimensions, where you want to ask questions such as:
//!  - Find the [nearest_n](`float::kdtree::KdTree::nearest_n`) item(s) to a query point, ordered by distance;
//!  - Find all items [within](`float::kdtree::KdTree::within`) a specified radius of a query point;
//!  - Find the ["best" n item(s) within](`float::kdtree::KdTree::best_n_within`) a specified distance of a query point, for some definition of "best"
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "3.0.0-beta.1"
//! ```
//!
//! ## Usage
//! ```rust
//! use kiddo::KdTree;
//! use kiddo::float::distance::SquaredEuclidean;
//! use kiddo::nearest_neighbour::NearestNeighbour;
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

#[macro_use]
extern crate doc_comment;

pub mod best_neighbour;
#[doc(hidden)]
pub(crate) mod common;
#[cfg(feature = "serialize")]
mod custom_serde;
pub mod distance_metric;
pub mod fixed;
pub mod float;
#[cfg(feature = "immutable")]
pub mod immutable;
mod mirror_select_nth_unstable_by;
pub mod nearest_neighbour;
#[doc(hidden)]
pub mod test_utils;
pub mod types;
pub mod within_unsorted_iter;

/// A floating-point k-d tree with default parameters.
///
/// `A` is the floating point type (`f32` or `f64`).
/// `K` is the number of dimensions. See [`float::kdtree::KdTree`] for details of how to use.
///
/// To manually specify more advanced parameters, use [`float::kdtree::KdTree`] directly.
/// To store positions using integer or fixed-point types, use [`fixed::kdtree::KdTree`].
pub type KdTree<A, const K: usize> = float::kdtree::KdTree<A, usize, K, 32, u32>;
