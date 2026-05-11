#![cfg_attr(
    all(kiddo_nightly, target_arch = "aarch64"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(
    all(kiddo_nightly, feature = "simd"),
    feature(target_feature_inline_always)
)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
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
//! A high-performance k-d tree library for exact and approximate nearest-neighbour
//! queries in low-dimensional spaces.
//!
//! Built with an aggressive focus on query performance, including cache-aware
//! layouts and optional SIMD-accelerated code paths. See the companion
//! benchmarking site to compare Kiddo against other k-d tree implementations
//! across a range of workloads.
//!
//! Kiddo v6 provides a single generic [`KdTree`] that supports floating-point
//! (`f64`, `f32`, `f16`), selected fixed-point (via the `fixed` crate), and
//! unsigned-integer (`u8`, `u16`, `u32`) types as coordinates, along with both mutable
//! and immutable usage patterns.
//!
//! Kiddo is designed for low-dimensional search problems, especially 2D, 3D,
//! and 4D workloads. Typical use cases include point-cloud analysis,
//! astronomical catalogue crossmatching, colour quantization and palette
//! lookup, local neighbourhood queries in simulations, and other
//! nearest-neighbour and radius-search tasks. Kiddo has been used for diverse
//! geographical and scientific workloads including geocoding, astronomy,
//! cosmology, computer-aided drug discovery, crystallography, and
//! computational neuroscience.
//!
//! If your points are known up front and the tree will be built once and then
//! queried, start with [`ImmutableKdTree`]. It offers the best query
//! performance and pairs well with `rkyv` for zero-copy loading of prebuilt
//! trees from disk; when used with memory-mapped files, loading can be
//! effectively instant.
//!
//! If you need to add or remove points after construction, start with
//! [`MutableKdTree`]. Mutable trees remain a good fit for many dynamic
//! workloads, but they do not currently perform dynamic rebalancing, so
//! workloads with substantial growth or heavy churn may benefit from periodic
//! rebuilds.
//!
//! [`ImmutableKdTree`] and [`MutableKdTree`] are convenience aliases for
//! [`KdTree`] with sensible defaults for these common read-heavy and mutable
//! workloads.
//!
//! Kiddo is not intended as a library for high-dimensional vector search or
//! feature matching over hundreds or thousands of dimensions, where plain
//! k-d trees are usually the wrong data structure and other approaches are more
//! appropriate. The API does not impose a hard dimensional limit, but Kiddo is
//! primarily intended for low-dimensional workloads.
//!
//! Kiddo supports the following query types:
//!
//! - [`KdTree::nearest_one`] finds the single nearest item to a query point.
//!   Useful for tasks like finding the nearest airport to a given location, or
//!   finding the nearest catalogued star to a sky position.
//!
//! - [`KdTree::best_n_within`] finds the "best" `n` items within a specified
//!   distance of a query point, for some definition of "best".
//!   For example, "give me the 5 largest settlements within 50km of a given
//!   point, ordered by descending population", or "the 5 brightest stars
//!   within a degree of a point on the sky, ordered brightest first".
//!
//! - [`KdTree::approx_nearest_one`] performs approximate nearest-neighbour
//!   (ANN) search, returning a good approximate nearest item, often much faster
//!   than exact nearest-neighbour search.
//!   Useful for latency-sensitive workloads like interactive point-cloud
//!   picking, or mapping image pixels to a palette colour during colour
//!   quantization.
//!
//! - [`KdTree::nearest_n`] performs k-nearest-neighbour (k-NN) search, finding
//!   the `n` nearest items to a query point ordered by distance.
//!   Useful for finding the nearest weather stations or sensors to a location,
//!   or generating candidate correspondences for point-cloud registration.
//!
//! - [`KdTree::nearest_n_within`] finds up to `n` nearest items within a
//!   specified radius of a query point, ordered by distance.
//!   Useful when you want the closest local neighbours inside a meaningful
//!   cutoff, such as the nearest shops within 5 miles, or nearby atoms within
//!   an interaction radius.
//!
//! - [`KdTree::within`] finds all items within a specified radius of a query
//!   point, ordered by distance.
//!   Useful for radial catalogue searches in astronomy, or collision and
//!   proximity queries where the full neighbourhood is needed in sorted order.
//!
//! - [`KdTree::within_unsorted`] finds all items within a specified radius of a
//!   query point without sorting the results.
//!   This is often faster than [`KdTree::within`] when result order does not
//!   matter, such as finding all customers within 5 miles of a store, or
//!   collecting point-cloud neighbourhoods for clustering or normal estimation.
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "6.0.0-alpha.1"
//! ```
//!
//! ## Usage
//! ```rust
//! use std::num::NonZero;
//!
//! use kiddo::leaf_strategy::VecOfArrays;
//! use kiddo::SquaredEuclidean;
//! use kiddo::NearestNeighbour;
//! use kiddo::{Eytzinger, ImmutableKdTree};
//!
//! let entries = vec![
//!     [0f64, 0f64],
//!     [1f64, 1f64],
//!     [2f64, 2f64],
//!     [3f64, 3f64]
//! ];
//!
//! let kdtree = ImmutableKdTree::new_from_slice(&entries);
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
//!
//! ## Optional Features
//!
//! Kiddo exposes a number of optional crate features:
//!
//! - `fixed` enables support for fixed-point coordinate
//!   types from the [`fixed`](https://docs.rs/fixed/latest/fixed) crate.
//!
//! - `tracing` **(enabled by default)** enables tracing-based instrumentation
//!   and logging.
//!
//! - `f16` enables support for half-precision floating-point coordinates via
//!   the [`half`](https://docs.rs/half/latest/half) crate.
//!
//! - `serde` enables serialization and deserialization via
//!   [`Serde`](https://docs.rs/serde/latest/serde/).
//!
//! - `rkyv_08` enables zero-copy serialization and deserialization via
//!   [`rkyv`](https://docs.rs/rkyv/latest/rkyv/) 0.8.x. This is particularly
//!   useful for prebuilt immutable trees that you want to load very quickly,
//!   especially in conjunction with memory-mapped files.
//!
//! - `simd` **(NIGHTLY)** enables handwritten SIMD and prefetch intrinsics for
//!   additional performance where available. This requires a nightly Rust
//!   toolchain.
//!
//! - `huge_pages` enables Linux-specific huge-page advice helpers for owned and
//!   archived tree storage.
//!
//! - `leaf_nta_prefetch` enables additional non-temporal leaf prefetch hints in
//!   some query paths. This is an advanced tuning feature and is only useful in
//!   specific workloads.
//!
//! Kiddo also contains a number of additional feature flags used for internal
//! experimentation, benchmarking, simulation, and specialized tuning. Most
//! users will not need them.
//!
//! **NOTE**: Support for rkyv 0.7 was removed in Kiddo v6.

// #[doc(hidden)]
// #[cfg(feature = "serde")]
// #[doc(hidden)]
// mod custom_serde;

/// Distance metrics
pub mod dist;

pub use crate::dist::{DotProduct, Manhattan, SquaredEuclidean};

pub mod huge_pages;

pub mod kd_tree;
pub use kd_tree::KdTree;

/// Convenience type alias for recommended default params for an immutable KdTree
pub type ImmutableKdTree<AX, const K: usize> =
    KdTree<AX, u32, Eytzinger<K>, VecOfArenas<AX, u32, K, 32>, K, 32>;

/// Convenience type alias for recommended default params for a mutable KdTree
pub type MutableKdTree<AX, const K: usize> =
    KdTree<AX, u32, Eytzinger<K>, VecOfArrays<AX, u32, K, 32>, K, 32>;

/// Leaf storage strategies for the kd-tree
pub mod leaf_strategy;
pub use leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};

/// Leaf view abstraction for accessing leaf data
pub mod leaf_view;

/// Chunked Leaf view abstraction for accessing leaf data
pub(crate) mod leaf_view_chunked;

mod mirror_select_nth_unstable_by;

/// Structs that are returned as query results
pub mod results;
pub use results::{best_neighbour::BestNeighbour, nearest_neighbour::NearestNeighbour};

#[cfg(feature = "rkyv_08")]
mod rkyv;

/// Stem Orderings
pub mod stem_strategy;
pub use stem_strategy::{Donnelly, DonnellyMarkerPf, DonnellyMarkerSimd, Eytzinger, EytzingerPf};

#[doc(hidden)]
#[cfg(feature = "test_utils")]
pub mod test_utils;

pub mod traits;
pub use traits::{
    axis::Axis, content::Content, distance_metric::DistanceMetric, leaf_strategy::LeafStrategy,
    stem_strategy::StemStrategy,
};
