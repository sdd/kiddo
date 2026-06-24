/// Donnelly stem ordering strategies and shared Donnelly internals.
pub mod donnelly;

/// Eytzinger Stem Ordering
#[doc(hidden)]
pub mod eytzinger;

mod block_size;
mod prefetch;

#[doc(hidden)]
pub use donnelly::simd_full::{CompareBlock3, CompareBlock4, SimdPrune, SimdSelectBestChildBlock3};

#[doc(inline)]
pub use donnelly::Donnelly;
#[doc(inline)]
pub use donnelly::{
    DonnellyNoPf, DonnellySimdDescent, DonnellySimdFull, DonnellyUnrolled, DonnellyUnrolledBlockDim,
};

#[doc(inline)]
pub use eytzinger::Eytzinger;
#[doc(inline)]
pub use eytzinger::EytzingerNoPf;

/// Marker types used to parameterize block-based stem strategies.
pub mod markers {
    #[doc(inline)]
    pub use super::block_size::BlockHeightMarker;
    #[doc(inline)]
    pub use super::block_size::{Block2, Block3, Block4, Block5, Block6, Block7};
}

#[doc(hidden)]
pub use block_size::BlockHeightMarker;
#[doc(hidden)]
pub use block_size::{Block2, Block3, Block4, Block5, Block6, Block7};

// TODO: modified Donnelly Block stem strategy that pads to the point where
//       the bottom row of the last block is free. Then, use that bottom row
//       to store the offset and leaf size for each left and right child of
//       the level above, eliminating the need for extents to be stored in the
//       leaf strategy, and avoiding the need for one layer of indirection when
//       retrieving the leaf offset
