#![feature(int_log)]
#![feature(int_roundings)]
#![feature(min_specialization)]
#![feature(stdsimd)]
#![feature(strict_provenance)]
#![feature(maybe_uninit_slice)]
#![doc(html_root_url = "https://docs.rs/sok/0.0.1")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/sok/issues/")]

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
pub mod test_utils;
pub mod types;

// pub use crate::sok::KdTree;
