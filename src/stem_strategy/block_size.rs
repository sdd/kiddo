/// Marker trait for block sizes used in Donnelly-style strategies
pub trait BlockHeightMarker: Copy + Clone + Send + Sync + 'static {
    /// The block height (number of levels per minor triangle)
    const BLOCK_HEIGHT: u32;
}

/// Marker type indicating a block size of 2.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block2;

/// Marker type indicating a block size of 3.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block3;

/// Marker type indicating a block size of 4.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block4;

/// Marker type indicating a block size of 5.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block5;

/// Marker type indicating a block size of 6.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block6;

/// Marker type indicating a block size of 7.
#[derive(Copy, Clone, Debug, Default)]
pub struct Block7;

impl BlockHeightMarker for Block2 {
    const BLOCK_HEIGHT: u32 = 2;
}
impl BlockHeightMarker for Block3 {
    const BLOCK_HEIGHT: u32 = 3;
}
impl BlockHeightMarker for Block4 {
    const BLOCK_HEIGHT: u32 = 4;
}
impl BlockHeightMarker for Block5 {
    const BLOCK_HEIGHT: u32 = 5;
}
impl BlockHeightMarker for Block6 {
    const BLOCK_HEIGHT: u32 = 6;
}
impl BlockHeightMarker for Block7 {
    const BLOCK_HEIGHT: u32 = 7;
}
