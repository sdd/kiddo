use aligned_vec::AVec;

/// Immutable leaf storage using chunk-tiled column-major points with a separate item array.
///
/// Intended layout:
/// - `point_arena` stores point coordinates in descending tile sizes (`32, 8, 4, 2, 1`)
/// - each tile is laid out column-major as `[x_tile][y_tile][z_tile]...`
/// - `items` stores the corresponding tile item ranges separately from the point arena
///
/// This is a TODO placeholder as a reminder about the alternate arena design.
#[allow(unused)]
pub struct VecOfStructOfTiles<A, T, const K: usize, const B: usize> {
    leaf_extents: Vec<(usize, usize)>,
    point_arena: AVec<A>,
    items: AVec<T>,
    size: usize,
}
