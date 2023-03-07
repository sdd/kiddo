#![feature(int_roundings)]
#![feature(min_specialization)]
#![feature(stdsimd)]
#![feature(strict_provenance)]
#![feature(maybe_uninit_slice)]
#![doc(html_root_url = "https://docs.rs/kiddo/2.0.0-beta.2")]
#![feature(rustdoc_missing_doc_code_examples)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![doc(html_root_url = "https://docs.rs/kiddo/2.0.0-beta.2")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]

//! # Kiddo
//!
//! A flexible, high-performance kd-tree library.
//!
//! v2 is a complete rewrite from the ground up, with a new internal architecture for
//! much-improved performance. As well as the existing floating point capability, Kiddo v2
//! now also supports fixed point / integers via the Fixed library. v2 also supports instant
//! zero-copy serialization and deserialization via Rkyv, as well as Serde.
//!
//! Kiddo is ideal for super-fast spatial / geospatial lookups and nearest-neighbour / KNN
//! queries for low-ish numbers of dimensions.
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "2.0.0-beta.2"
//! ```
//!
//! ## Usage
//! ```rust
//! use kiddo::KdTree;
//! use kiddo::distance::squared_euclidean;
//! use kiddo::types::Index;
//!
//! let a: ([f64; 2], u32) = ([0f64, 0f64], 0);
//! let b: ([f64; 2], u32) = ([1f64, 1f64], 1);
//! let c: ([f64; 2], u32) = ([2f64, 2f64], 2);
//! let d: ([f64; 2], u32) = ([3f64, 3f64], 3);
//!
//! let mut kdtree: KdTree<f64, u32, 2, 32, u32> = KdTree::new();
//!
//! kdtree.add(&a.0, a.1);
//! kdtree.add(&b.0, b.1);
//! kdtree.add(&c.0, c.1);
//! kdtree.add(&d.0, d.1);
//!
//! assert_eq!(kdtree.size(), 4);
//! assert_eq!(
//!     kdtree.nearest_n(&a.0, 1, &squared_euclidean).collect::<Vec<_>>(),
//!     vec![(0f64, 0)]
//! );
//! assert_eq!(
//!     kdtree.nearest_n(&a.0, 2, &squared_euclidean).collect::<Vec<_>>(),
//!     vec![(0f64, 0), (2f64, 1)]
//! );
//! assert_eq!(
//!     kdtree.nearest_n(&a.0, 3, &squared_euclidean).collect::<Vec<_>>(),
//!     vec![(0f64, 0), (2f64, 1), (8f64, 2)]
//! );
//! ```

#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(feature = "serialize")]
extern crate serde_derive;

#[cfg(feature = "serialize")]
mod custom_serde;
pub mod distance;

pub mod fixed;
pub mod float;
mod mirror_select_nth_unstable_by;
#[doc(hidden)]
pub mod test_utils;
pub mod types;

pub use crate::fixed::kdtree::KdTree as FixedKdTree;
pub use crate::float::kdtree::KdTree;
