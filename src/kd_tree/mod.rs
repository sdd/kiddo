//! Flexible kd-trees that can be used with float or fixed point, mutable or immutable, and selectable stem ordering strategies

mod construction;
/// Leaf storage strategies for the kd-tree
pub mod leaf_strategies;
/// Leaf view abstraction for accessing leaf data
pub mod leaf_view;
mod query;
mod query_orchestrator;
mod query_stack;
mod result_collection;
mod traits;

use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;
use aligned_vec::{AVec, CACHELINE_ALIGN};
use std::num::NonZeroUsize;

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
        stems_depth: usize,
        leaf_count: usize,
    },
    /// Mutable strategies in pristine state: no structural mutations yet.
    ///
    /// Uses arithmetic resolution like `Arithmetic`, but can transition
    /// to `Mapped` when the first leaf split/merge occurs.
    Pristine {
        stems_depth: usize,
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
        leaf_idx_map: Vec<Option<NonZeroUsize>>,
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
    stem_leaf_resolution: StemLeafResolution,

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
        Self {
            stems: AVec::new(CACHELINE_ALIGN),
            leaves: LS::new_with_empty_leaf(),
            stem_leaf_resolution: StemLeafResolution::Arithmetic {
                stems_depth: 0,
                leaf_count: 0,
            },
            size: 0,
            max_stem_level: -1,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B>, // + Default,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_tree::leaf_strategies::dummy::DummyLeafStrategy;
    use crate::Eytzinger;

    #[test]
    fn test_default() {
        let kd_tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> = Default::default();

        assert_eq!(kd_tree.size, 0);
        assert_eq!(kd_tree.max_stem_level, 0);
        assert!(kd_tree.is_empty());
    }

    #[test]
    fn test_from_iterator_empty() {
        let points = vec![[0.0f64; 3]];

        let kd_tree: KdTree<f64, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> =
            points.into_iter().enumerate().collect();

        assert_eq!(kd_tree.size, 0);
    }
}
