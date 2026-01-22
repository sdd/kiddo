/// Donnelly Stem Ordering
// pub mod donnelly_1;
pub mod donnelly_2;
pub mod donnelly_2_pf;
pub mod donnelly_3;
// pub mod donnelly_4;
// pub mod donnelly_5;

/// Eytzinger Stem Ordering
pub mod eytzinger;
pub mod eytzinger_pf;
pub mod eytzinger_pf_far;

mod block_size;
mod donnelly_2_pf_blockmarker;

/// Donnelly Stem Ordering core logic
pub mod donnelly_core;
mod prefetch;

mod donnelly_2_blockmarker_simd;

pub use donnelly_2_blockmarker_simd::{
    CompareBlock3, CompareBlock4, DonnellyMarkerSimd, SimdPrune,
};

pub use donnelly_2::Donnelly;
pub use donnelly_2_pf_blockmarker::DonnellyMarkerPf;
pub use donnelly_3::DonnellySwPre;
pub use eytzinger::Eytzinger;
pub use eytzinger_pf::EytzingerPf;

pub use block_size::BlockSizeMarker;
pub use block_size::{Block3, Block4, Block5, Block6, Block7};
