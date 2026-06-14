/// Donnelly Stem Ordering
#[doc(hidden)]
pub mod donnelly_2;
#[doc(hidden)]
pub mod donnelly_2_pf;
#[doc(hidden)]
pub mod donnelly_3;
#[doc(hidden)]
pub mod donnelly_simd_descent;

/// Eytzinger Stem Ordering
#[doc(hidden)]
pub mod eytzinger;
#[doc(hidden)]
pub mod eytzinger_pf;
#[doc(hidden)]
pub mod eytzinger_pf_far;

mod block_size;
mod donnelly_2_pf_blockmarker;

/// Donnelly Stem Ordering core logic
#[doc(hidden)]
pub mod donnelly_core;
mod prefetch;

#[doc(hidden)]
pub mod donnelly_2_blockmarker_simd;

#[doc(hidden)]
pub use donnelly_2_blockmarker_simd::{
    CompareBlock3, CompareBlock4, DistanceMetricSimdBlock3, DistanceMetricSimdBlock4,
    DonnellyMarkerSimd, SimdPrune, SimdSelectBestChildBlock3,
};

#[doc(inline)]
pub use donnelly_2::Donnelly;
#[doc(inline)]
pub use donnelly_2_pf_blockmarker::{DonnellyMarkerPf, DonnellyMarkerScalar};
#[doc(inline)]
pub use donnelly_3::DonnellySwPre;
#[doc(inline)]
pub use donnelly_simd_descent::DonnellySimdDescent;
#[doc(inline)]
pub use eytzinger::Eytzinger;
#[doc(inline)]
pub use eytzinger_pf::EytzingerPf;

/// Marker types used to parameterize block-based stem strategies.
pub mod markers {
    #[doc(inline)]
    pub use super::block_size::BlockSizeMarker;
    #[doc(inline)]
    pub use super::block_size::{Block3, Block4, Block5, Block6, Block7};
}

#[doc(hidden)]
pub use block_size::BlockSizeMarker;
#[doc(hidden)]
pub use block_size::{Block3, Block4, Block5, Block6, Block7};

// TODO: modified Donnelly Block stem strategy that pads to the point where
//       the bottom row of the last block is free. Then, use that bottom row
//       to store the offset and leaf size for each left and right child of
//       the level above, eliminating the need for extents to be stored in the
//       leaf strategy, and avoiding the need for one layer of indirection when
//       retrieving the leaf offset
