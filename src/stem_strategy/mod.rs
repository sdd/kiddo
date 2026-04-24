/// Donnelly Stem Ordering
pub mod donnelly_2;
pub mod donnelly_2_pf;
pub mod donnelly_3;
pub mod donnelly_simd_descent;

/// Eytzinger Stem Ordering
pub mod eytzinger;
pub mod eytzinger_pf;
pub mod eytzinger_pf_far;

mod block_size;
mod donnelly_2_pf_blockmarker;

/// Donnelly Stem Ordering core logic
pub mod donnelly_core;
mod prefetch;

pub mod donnelly_2_blockmarker_simd;

pub use donnelly_2_blockmarker_simd::{
    CompareBlock3, CompareBlock4, DistanceMetricSimdBlock3, DistanceMetricSimdBlock4,
    DonnellyMarkerSimd, SimdPrune, SimdSelectBestChildBlock3,
};

pub use donnelly_2::Donnelly;
pub use donnelly_2_pf_blockmarker::{DonnellyMarkerPf, DonnellyMarkerScalar};
pub use donnelly_3::DonnellySwPre;
pub use donnelly_simd_descent::DonnellySimdDescent;
pub use eytzinger::Eytzinger;
pub use eytzinger_pf::EytzingerPf;

pub use block_size::BlockSizeMarker;
pub use block_size::{Block3, Block4, Block5, Block6, Block7};

// TODO: modified Donnelly Block stem strategy that pads to the point where
//       the bottom row of the last block is free. Then, use that bottom row
//       to store the offset and leaf size for each left and right child of
//       the level above, eliminating the need for extents to be stored in the
//       leaf strategy, and avoiding the need for one layer of indirection when
//       retrieving the leaf offset
