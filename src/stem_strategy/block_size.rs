/// Marker trait for block sizes used in Donnelly-style strategies
pub trait BlockSizeMarker: Copy + Clone + Send + Sync + 'static {
    /// The block size (number of levels per minor triangle)
    const SIZE: usize;
}

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

impl BlockSizeMarker for Block3 {
    const SIZE: usize = 3;
}
impl BlockSizeMarker for Block4 {
    const SIZE: usize = 4;
}
impl BlockSizeMarker for Block5 {
    const SIZE: usize = 5;
}
impl BlockSizeMarker for Block6 {
    const SIZE: usize = 6;
}
impl BlockSizeMarker for Block7 {
    const SIZE: usize = 7;
}
