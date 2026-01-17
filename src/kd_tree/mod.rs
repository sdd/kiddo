//! Flexible kd-trees that can be used with float or fixed point, mutable or immutable, and selectable stem ordering strategies

mod construction;
/// Leaf storage strategies for the kd-tree
pub mod leaf_strategies;
/// Leaf view abstraction for accessing leaf data
pub mod leaf_view;
mod query;
mod query_orchestrator;
pub(crate) mod query_stack;
pub(crate) mod query_stack_simd;
mod result_collection;
pub(crate) mod traits;

use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;
use aligned_vec::{AVec, CACHELINE_ALIGN};
use nonmax::NonMaxUsize;

/// Strategy for resolving stem indices to leaf indices during traversal.
///
/// Different variants optimize for different tree usage patterns:
/// - `Arithmetic`: For immutable trees where all leaves are at the same depth
/// - `Pristine`: For mutable trees that haven't had structural mutations yet
/// - `Mapped`: For mutable trees after leaf splits/merges have occurred
#[derive(Clone, Debug, PartialEq)]
pub enum StemLeafResolution {
    /// Immutable strategies: leaf index can be calculated arithmetically.
    ///
    /// All leaves are guaranteed to be at the same depth, so leaf indices
    /// can be computed directly from stem indices.
    Arithmetic {
        /// how many levels deep the stem tree is
        stems_depth: usize,
        /// how many leaves there are
        leaf_count: usize,
    },
    /// Mutable strategies in pristine state: no structural mutations yet.
    ///
    /// Uses arithmetic resolution like `Arithmetic`, but can transition
    /// to `Mapped` when the first leaf split/merge occurs.
    Pristine {
        /// initial stem depth
        stems_depth: usize,
        /// how many leaves there are initially
        leaf_count: usize,
    },
    /// Mutable strategies after structural mutations (split/merge).
    ///
    /// Requires explicit mapping from terminal stem indices to leaf indices
    /// because leaves may be at different depths.
    Mapped {
        /// Index of the first stem that might point to a leaf
        min_stem_leaf_idx: usize,
        /// Maps stem indices to leaf indices.
        /// `None` means the stem has children, `Some(idx)` means it points to leaf `idx`.
        leaf_idx_map: Vec<Option<NonMaxUsize>>,
    },
}

impl StemLeafResolution {
    /// Returns true if this resolution strategy uses arithmetic (fast path).
    #[inline(always)]
    pub fn uses_arithmetic(&self) -> bool {
        matches!(self, Self::Arithmetic { .. } | Self::Pristine { .. })
    }
}

/// A k-d tree for efficient spatial queries.
///
/// # Type Parameters
/// * `A` - Axis/coordinate type (e.g., f32, f64, or fixed-point types)
/// * `T` - Content/item type stored at each point
/// * `SS` - Stem ordering strategy (e.g., Eytzinger, Donnelly)
/// * `LS` - Leaf storage strategy
/// * `K` - Dimensionality (number of dimensions)
/// * `B` - Bucket size (maximum items per leaf node)
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<
    A,              // Axis
    T,              // Content,
    SS,             // StemStrategy
    LS,             // LeafStrategy
    const K: usize, // dimensionality
    const B: usize, // bucket size
> {
    stems: AVec<A>,
    leaves: LS,
    pub(crate) stem_leaf_resolution: StemLeafResolution,

    size: usize,
    max_stem_level: i32,
    pub(crate) _phantom: std::marker::PhantomData<(SS, T)>,
}

impl<A, T, SS, LS, const K: usize, const B: usize> Default for KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    fn default() -> Self {
        use crate::traits_unified_2::Mutability;

        // For mutable trees, initialize with sentinel stem at root
        let (stems, max_stem_level, stem_leaf_resolution) = if LS::Mutability::is_mutable() {
            // Get the root index for this stem strategy
            let root_idx = SS::new_no_ptr().stem_idx();

            // Create stems array with sentinel value at root
            let mut stems = AVec::new(CACHELINE_ALIGN);
            stems.resize(root_idx + 1, A::max_value());

            // Start in Mapped state - map root directly to the single initial leaf
            let mut leaf_idx_map = vec![None; root_idx + 1];
            leaf_idx_map[root_idx] = NonMaxUsize::new(0);

            let stem_leaf_resolution = crate::kd_tree::StemLeafResolution::Mapped {
                min_stem_leaf_idx: 0,
                leaf_idx_map,
            };

            (stems, 0, stem_leaf_resolution)
        } else {
            // Immutable trees start empty
            let stems = AVec::new(CACHELINE_ALIGN);
            let stem_leaf_resolution = StemLeafResolution::Arithmetic {
                stems_depth: 0,
                leaf_count: 0,
            };

            (stems, -1, stem_leaf_resolution)
        };

        Self {
            stems,
            leaves: LS::new_with_empty_leaf(),
            stem_leaf_resolution,
            size: 0,
            max_stem_level,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Returns `true` if the tree contains no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of points in the tree.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the maximum stem level in the tree.
    #[inline]
    pub fn max_stem_level(&self) -> i32 {
        self.max_stem_level
    }

    /// Returns the number of leaf nodes in the tree.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.leaves.leaf_count()
    }

    /// Find which leaf contains a specific item.
    /// Returns `Some((leaf_idx, position_in_leaf))` if found, `None` if not found.
    pub fn find_leaf_for_item(&self, target_item: T) -> Option<(usize, usize)>
    where
        T: PartialEq,
    {
        for leaf_idx in 0..self.leaves.leaf_count() {
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            let (_points, items) = leaf_view.into_parts();

            for (pos_in_leaf, item) in items.iter().enumerate() {
                if *item == target_item {
                    return Some((leaf_idx, pos_in_leaf));
                }
            }
        }
        None
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> FromIterator<(usize, [A; K])>
    for KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B> + Default,
    SS: StemStrategy,
{
    fn from_iter<I: IntoIterator<Item = (usize, [A; K])>>(_iter: I) -> Self {
        // TODO: Proper impl
        Self::default()
    }
}

// Display implementation for debugging
impl<A, T, SS, LS, const K: usize, const B: usize> std::fmt::Display for KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + std::fmt::Display,
    T: Basics + std::fmt::Display,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "KdTree {{")?;
        writeln!(f, "  Summary:")?;
        writeln!(f, "    size: {}", self.size)?;
        writeln!(f, "    max_stem_level: {}", self.max_stem_level)?;
        writeln!(f, "    stem len: {}", self.stems.len())?;
        writeln!(f, "    leaf count: {}", self.leaves.leaf_count())?;
        writeln!(f)?;

        // Display stems array
        writeln!(f, "  Stems (len={}):", self.stems.len())?;
        writeln!(f, "    [")?;
        for (i, stem) in self.stems.iter().enumerate() {
            if i % 8 == 0 {
                write!(f, "     ")?;
            }
            write!(f, "{:8.3}", stem)?;
            if i < self.stems.len() - 1 {
                write!(f, ",")?;
            }
            if (i + 1) % 8 == 0 || i == self.stems.len() - 1 {
                writeln!(f)?;
            } else {
                write!(f, "\t")?;
            }
        }
        writeln!(f, "    ]")?;
        writeln!(f)?;

        // Display stem_leaf_resolution
        writeln!(f, "  StemLeafResolution:")?;
        match &self.stem_leaf_resolution {
            StemLeafResolution::Arithmetic {
                stems_depth,
                leaf_count,
            } => {
                writeln!(f, "    Arithmetic {{")?;
                writeln!(f, "      stems_depth: {}", stems_depth)?;
                writeln!(f, "      leaf_count: {}", leaf_count)?;
                writeln!(f, "    }}")?;
            }
            StemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } => {
                writeln!(f, "    Pristine {{")?;
                writeln!(f, "      stems_depth: {}", stems_depth)?;
                writeln!(f, "      leaf_count: {}", leaf_count)?;
                writeln!(f, "    }}")?;
            }
            StemLeafResolution::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => {
                writeln!(f, "    Mapped {{")?;
                writeln!(f, "      min_stem_leaf_idx: {}", min_stem_leaf_idx)?;
                writeln!(f, "      leaf_idx_map (len={}): [", leaf_idx_map.len())?;
                for (i, entry) in leaf_idx_map.iter().enumerate() {
                    match entry {
                        Some(idx) => writeln!(f, "        {}: Some({})", i, idx)?,
                        None => writeln!(f, "        {}: None", i)?,
                    }
                }
                writeln!(f, "      ]")?;
                writeln!(f, "    }}")?;
            }
        }
        writeln!(f)?;

        // Display leaves
        writeln!(f, "  Leaves (count={}):", self.leaves.leaf_count())?;
        for leaf_idx in 0..self.leaves.leaf_count() {
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            let (points, items) = leaf_view.into_parts();

            write!(f, "    Leaf {} (count={}): [", leaf_idx, items.len())?;
            for i in 0..items.len() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "(")?;
                for dim in 0..K {
                    if dim > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.3}", points[dim][i])?;
                }
                write!(f, "): {}", items[i])?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_tree::leaf_strategies::dummy::DummyLeafStrategy;
    use crate::kd_tree::leaf_strategies::FlatVec;
    use crate::stem_strategies::Donnelly;
    use crate::Eytzinger;

    #[test]
    fn test_default() {
        let kd_tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> = Default::default();

        assert_eq!(kd_tree.size, 0);
        assert!(kd_tree.is_empty());
    }

    #[test]
    fn test_from_iterator_empty() {
        let points = vec![[0.0f64; 3]];

        let kd_tree: KdTree<f64, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> =
            points.into_iter().enumerate().collect();

        assert_eq!(kd_tree.size, 0);
    }

    #[test]
    fn test_stem_height_padding_donnelly_l4() {
        // Create a tree with a height that needs padding to block boundary
        // With 100 items and bucket size 32, we get 4 leaves
        // 4 leaves -> depth = log2(4) = 2 levels (levels 0, 1)
        // max_stem_level = 1 (0-indexed)
        // stems_depth = max_stem_level + 1 = 2
        // For Donnelly<4>, block_size = 4
        // 2 % 4 = 2, so we need padding of 4 - 2 = 2 levels
        // Final depth should be 4, max_stem_level should be 3

        const TREE_SIZE: usize = 100;
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE)
            .map(|i| {
                let x = (i as f32) / (TREE_SIZE as f32);
                [x, x * 2.0, x * 3.0, x * 4.0]
            })
            .collect();

        let tree: KdTree<f32, u32, Donnelly<4, 64, 4, 4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        // Verify padding was applied
        let stems_depth = tree.max_stem_level() + 1;
        let block_size = 4;
        assert_eq!(
            stems_depth % block_size,
            0,
            "Stem tree depth should be a multiple of block size. depth={}, block_size={}",
            stems_depth,
            block_size
        );

        // With 100 items and bucket size 32, we have 4 leaves
        // Natural depth would be 2 (log2(4) = 2)
        // Padded to block size 4, should be 4
        assert_eq!(
            stems_depth, 4,
            "Expected padded depth of 4 for tree with 4 leaves and block size 4"
        );
        assert_eq!(
            tree.max_stem_level(),
            3,
            "Expected max_stem_level of 3 (depth 4 - 1)"
        );

        // Verify tree still works correctly with padding
        let query_point = [0.5f32, 1.0f32, 1.5f32, 2.0f32];
        let leaf_idx = tree.get_leaf_idx(&query_point);
        assert!(
            leaf_idx < tree.leaf_count(),
            "Leaf index should be valid. leaf_idx={}, leaf_count={}",
            leaf_idx,
            tree.leaf_count()
        );
    }
}
