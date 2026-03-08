//! Definitions and implementations for some traits that are common between the [`float`](crate::mutable::float), [`immutable`](crate::immutable) and [`fixed`](crate::mutable::fixed)  modules
// use std::num::NonZero;
use crate::kd_tree::query_stack::StackTrait;
use crate::stem_strategies::donnelly_2_blockmarker_simd::backtrack_traits::{
    BacktrackBlock3, BacktrackBlock4,
};
use crate::traits_unified_2::AxisUnified;
// use crate::{BestNeighbour, NearestNeighbour, WithinUnsortedIter};
use aligned_vec::AVec;
use az::Cast;
use divrem::DivCeil;
use fixed::prelude::ToFixed;
use fixed::traits::Fixed;
use num_traits::float::FloatCore;
use num_traits::{PrimInt, Unsigned, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ptr::NonNull;

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on float `KdTree`s. This will be [`f64`] or [`f32`],
/// or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if used with
/// the [`half`](https://docs.rs/half/latest/half) crate
pub trait Axis:
    FloatCore + Default + Debug + Copy + Sync + Send + std::ops::AddAssign + Sum
{
    /// returns absolute diff between two values of a type implementing this trait
    fn saturating_dist(self, other: Self) -> Self;

    /// Used in query methods to update the rd value. A saturating add for Fixed and an add for Float
    fn rd_update(rd: Self, delta: Self) -> Self;
}

impl<T: FloatCore + Default + Debug + Copy + Sync + Send + std::ops::AddAssign + Sum> Axis for T {
    fn saturating_dist(self, other: Self) -> Self {
        (self - other).abs()
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn rd_update(rd: Self, delta: Self) -> Self {
        rd + delta
    }
}

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on [`FixedKdTree`](crate::mutable::fixed::kdtree::KdTree). A type from the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate will implement
/// all of the traits required by Axis. For example, [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html).
pub trait AxisFixed: Fixed + ToFixed + PartialOrd + Default + Debug + Copy + Sync + Send {
    /// used in query methods to update the rd value. Basically a saturating add for Fixed and an add for Float
    fn rd_update(rd: Self, delta: Self) -> Self;
}
impl<T: Fixed + ToFixed + PartialOrd + Default + Debug + Copy + Sync + Send> AxisFixed for T {
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn rd_update(rd: Self, delta: Self) -> Self {
        rd.saturating_add(delta)
    }
}

/// Content trait.
///
/// Must be implemented by any type that you want to use to represent the content
/// stored in a KdTree. Generally this will be `usize`, `u32`, or for trees with less
/// than 65,535 points, you could use a `u16`. All these types implement `Content` with no
/// extra changes. Start off with a `usize`, as that's easiest
/// since you won't need to cast to / from usize when using query results to index into
/// a Vec. Try switching to a smaller type and benchmarking to see if you get better
/// performance. Any type that satisfies these trait constraints may be used; in
/// particular, we use T::default() to initialize the KdTree content.
pub trait Content: PartialEq + Default + Clone + Copy + Ord + Debug + Sync + Send {}

impl<T: PartialEq + Default + Clone + Copy + Ord + Debug + Sync + Send> Content for T {}

/// Implemented on u16 and u32 so that they can be used internally to index the
/// `Vec`s of Stem and Leaf nodes.
///
/// Allows `u32` or `u16` to be used as the 5th generic parameter of `float::KdTree`
/// and `fixed::KdTree`. If you will be storing fewer than `BUCKET_SIZE` * ~32k items
/// in the tree, selecting `u16` will slightly reduce the size of the Stem Nodes,
/// ensuring that more of them can be kept in the CPU cache, which may improve
/// performance (this may be offset on some architectures if it results in a
/// misalignment penalty).
pub trait Index: PrimInt + Unsigned + Zero + Cast<usize> + Sync {
    #[doc(hidden)]
    type T: Cast<usize>;
    #[doc(hidden)]
    fn max() -> Self;
    #[doc(hidden)]
    fn min() -> Self;
    #[doc(hidden)]
    fn leaf_offset() -> Self;
    #[doc(hidden)]
    fn ilog2(self) -> Self;
    #[doc(hidden)]
    fn div_ceil(self, b: Self::T) -> Self;
    #[doc(hidden)]
    fn capacity_with_bucket_size(bucket_size: usize) -> usize;
}

impl Index for u32 {
    type T = u32;
    fn max() -> u32 {
        u32::MAX
    }
    fn min() -> u32 {
        0u32
    }
    fn leaf_offset() -> u32 {
        u32::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u32 {
        u32::ilog2(self)
    }
    fn div_ceil(self, b: u32) -> u32 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        ((u32::MAX - u32::MAX.overflowing_shr(1).0) as usize).saturating_mul(bucket_size)
    }
}

impl Index for u16 {
    type T = u16;
    fn max() -> u16 {
        u16::MAX
    }
    fn min() -> u16 {
        0u16
    }
    fn leaf_offset() -> u16 {
        u16::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u16 {
        u16::ilog2(self) as u16
    }
    fn div_ceil(self, b: u16) -> u16 {
        DivCeil::div_ceil(self, b)
    }
    fn capacity_with_bucket_size(bucket_size: usize) -> usize {
        ((u16::MAX - u16::MAX.overflowing_shr(1).0) as usize).saturating_mul(bucket_size)
    }
}

pub(crate) fn is_stem_index<IDX: Index<T = IDX>>(x: IDX) -> bool {
    x < <IDX as Index>::leaf_offset()
}

/// Trait that needs to be implemented by any potential distance
/// metric to be used within queries
pub trait DistanceMetric<A, const K: usize> {
    /// returns the distance between two K-d points, as measured
    /// by a particular distance metric
    fn dist(a: &[A; K], b: &[A; K]) -> A;

    /// returns the distance between two points along a single axis,
    /// as measured by a particular distance metric.
    ///
    /// (needs to be implemented as it is used by the NN query implementations
    /// to extend the minimum acceptable distance for a node when recursing
    /// back up the tree)
    fn dist1(a: A, b: A) -> A;
}

/// Trait that needs to be implemented by any potential distance
/// metric to be used within queries on fixed-point trees
pub trait DistanceMetricFixed<A, const K: usize, R = A> {
    /// returns the distance between two K-d points, as measured
    /// by a particular distance metric
    fn dist(a: &[A; K], b: &[A; K]) -> R;

    /// returns the distance between two points along a single axis,
    /// as measured by a particular distance metric.
    ///
    /// (needs to be implemented as it is used by the NN query implementations
    /// to extend the minimum acceptable distance for a node when recursing
    /// back up the tree)
    fn dist1(a: A, b: A) -> R;
}

/// Trait that needs to be implemented by any potential stem ordering
/// algorithm used by a KdTree.
pub trait StemStrategy: Clone + Sync + Send {
    /// The stem index of the root node of the tree
    const ROOT_IDX: usize = 0;

    /// The block size of this strategy
    ///
    /// The default is 1, which means that the strategy is not block-based.
    const BLOCK_SIZE: usize = 1;

    /// Compact state persisted on scalar backtracking stacks.
    ///
    /// Scalar strategies can use this to store only the state needed to resume a deferred branch.
    /// SIMD / block strategies may ignore this and continue to use custom stack types.
    type DeferredState: Sized;

    /// Query stack context type for backtracking queries.
    ///
    /// Non-block strategies use simple scalar stack context (QueryStackContext).
    /// Block-based SIMD strategies use SimdQueryStackContext.
    type StackContext<A>: Sized
    where
        Self: Sized;

    /// Query stack type for backtracking queries.
    ///
    /// Non-block strategies use simple scalar stack (QueryStack).
    /// Block-based SIMD strategies use SimdQueryStack.
    type Stack<A>: Default + crate::kd_tree::query_stack::StackTrait<A, Self>
    where
        Self: Sized;

    /// Create a new instance of this strategy at the root.
    fn new(stems_ptr: NonNull<u8>) -> Self;

    /// Create a new instance of this strategy at the root, with a dangling pointer.
    ///
    /// Useful for generating traversal indices without performing prefetches
    #[inline(always)]
    fn new_no_ptr() -> Self {
        Self::new(NonNull::dangling())
    }

    /// Returns the block size of this strategy
    #[inline(always)]
    fn block_size() -> usize {
        Self::BLOCK_SIZE
    }

    /// Get the current stem index this strategy points to.
    fn stem_idx(&self) -> usize;

    /// Snapshot the minimal scalar deferred state needed to resume traversal later.
    fn deferred_state(&self) -> Self::DeferredState;

    /// Restore this strategy from deferred scalar traversal state.
    ///
    /// Implementations may assume `self` already holds a valid `stems_ptr`.
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState);

    /// Get the current leaf index this strategy points to.
    fn leaf_idx(&self) -> usize;

    /// Get the current dimension (query time)
    fn dim(&self) -> usize;

    /// Get the current dimension (construction time)
    fn construction_dim(&self) -> usize {
        self.dim()
    }

    /// Get the current level
    fn level(&self) -> i32;

    /// Advance `self` down to a child in-place.
    fn traverse(&mut self, is_right: bool);

    /// Advance `self` down to a child in-place. Specialized for use as one
    /// of the non-final stages when loop-unrolling to the level of a minor tri height
    #[inline(always)]
    fn traverse_head(&mut self, is_right: bool) {
        self.traverse(is_right);
    }

    /// Advance `self` down to a child in-place. Specialized for use as the
    /// last stage when loop-unrolled to the level of a minor tri height
    #[inline(always)]
    fn traverse_tail(&mut self, is_right: bool) {
        self.traverse(is_right);
    }

    /// Advance `self` down to one child, returning the other.
    /// - `self` mutates into the left child
    /// - return value is the right child
    fn branch(&mut self) -> Self;

    /// Advance `self` to the "closer" child, returning the "further" one.
    #[inline(always)]
    fn branch_relative(&mut self, is_right: bool) -> Self {
        if is_right {
            let mut right = self.branch();
            std::mem::swap(self, &mut right);
            right
        } else {
            self.branch()
        }
    }

    /// Split `self` into two independent child strategies (left, right).
    fn split(mut self) -> (Self, Self)
    where
        Self: Sized,
    {
        let right = self.branch();
        (self, right)
    }

    /// Split `self` into (closer, further) given a direction.
    fn split_relative(self, is_right: bool) -> (Self, Self)
    where
        Self: Sized,
    {
        let (l, r) = self.split();
        if is_right {
            (r, l)
        } else {
            (l, r)
        }
    }

    /// Get the stem indices where the left and right children would be located.
    /// Returns (left_child_stem_idx, right_child_stem_idx).
    fn child_indices(&self) -> (usize, usize);

    /// Calculate the stem node count for a given leaf node count.
    fn get_stem_node_count_from_leaf_node_count(_leaf_node_count: usize) -> usize {
        unimplemented!()
    }

    /// Factor by which to pad the stem node allocation.
    fn stem_node_padding_factor() -> usize {
        1
    }

    /// Trim unneeded stem nodes.
    fn trim_unneeded_stems<A: AxisUnified<Coord = A>>(
        _stems: &mut AVec<A>,
        _max_stem_level: usize,
    ) {
        // Default: no-op
    }

    /// Emit cache-simulation events while advancing one level in the stem tree.
    ///
    /// Implementations should mirror `traverse` behavior but also report memory
    /// accesses via `event_tx` for the cache simulator.
    #[cfg(feature = "simulator")]
    fn simulate_traverse(
        &mut self,
        _is_right: bool,
        _event_tx: &std::sync::mpsc::Sender<crate::cache_simulator::Event>,
    ) {
        unimplemented!();
    }

    /// Get leaf index for a query point. Default uses simple while loop.
    /// Block-based strategies override with unrolled loops.
    fn get_leaf_idx<A: AxisUnified, const K: usize>(
        stems: &[A],
        query: &[A; K],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat = Self::new(stems_ptr);

        while stem_strat.level() <= max_stem_level {
            let pivot = unsafe { stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right = unsafe { *query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right);
        }

        stem_strat.leaf_idx()
    }

    /// Single step of backtracking traversal.
    ///
    /// Default implementation handles level-by-level traversal with one pivot per step.
    /// Block-based strategies override to handle multiple levels at once with SIMD.
    ///
    /// Returns true if traversal should continue (more levels to go), false if at leaf.
    #[inline(always)]
    fn backtracking_traverse_step<A, O, D, const K: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K],
        query_wide: &[O; K],
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: AxisUnified<Coord = A>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K, Output = O>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, O>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, O>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        // Default implementation for scalar strategies
        // SIMD strategies override this entire method
        if self.level() > max_stem_level {
            return false;
        }

        let pivot = *unsafe { stems.get_unchecked(self.stem_idx()) };

        if pivot < A::max_value() {
            let query_elem = *unsafe { query.get_unchecked(*dim) };
            let is_right_child = query_elem >= pivot;

            let old_stem_idx = self.stem_idx();
            let far_ctx = self.branch_relative(is_right_child);

            tracing::trace!(
                %pivot,
                dim = %self.dim(),
                %query_elem,
                %is_right_child,
                %old_stem_idx,
                new_stem_idx = %self.stem_idx(),
                level = self.level(),
                "Traverse down one level"
            );

            let pivot_wide: O = D::widen_coord(pivot);
            let query_elem_wide = *unsafe { query_wide.get_unchecked(*dim) };

            let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
            let old_off = *unsafe { off.get_unchecked(*dim) };

            let new_dist1 = D::dist1(new_off, O::zero());
            let old_dist1 = D::dist1(old_off, O::zero());
            let rd_far = O::saturating_add(rd - old_dist1, new_dist1);

            // Only push if the sibling is worth exploring
            if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                stack.push(
                    crate::kd_tree::query_stack::scalar_ctx_from_parts::<O, Self>(
                        far_ctx.deferred_state(),
                        new_off,
                        rd_far,
                    ),
                );
            }
        } else {
            self.traverse(false);
        }

        *dim = self.dim();
        true
    }

    /// Execute backtracking query with explicit stack.
    ///
    /// Default implementation delegates to KdTree's backtracking_query_with_stack_impl.
    /// Block-based SIMD strategies can override to use SIMD stack with block pruning.
    ///
    /// This method exists to allow strategies to customize backtracking behavior at compile time.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn backtracking_query_with_stack<A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &crate::kd_tree::KdTree<A, T, Self, LS, K2, B>,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(&crate::kd_tree::leaf_view::LeafView<A, T, K2, B>, &[O; K2], &mut QC),
    ) where
        Self: Sized,
        A: AxisUnified<Coord = A>,
        T: crate::traits_unified_2::Basics + Copy + Default + PartialOrd + PartialEq,
        O: AxisUnified<Coord = O>
            + crate::stem_strategies::SimdPrune
            + BacktrackBlock3
            + BacktrackBlock4,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K2, O>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K2, O>,
        QC: crate::kd_tree::traits::QueryContext<A, O, K2>,
        LS: crate::traits_unified_2::LeafStrategy<A, T, Self, K2, B>,
        Self::Stack<O>: StackTrait<O, Self>,
    {
        // Default implementation - delegates to KdTree's scalar implementation
        // Requires Stack = QueryStack. SIMD strategies must fully override this method.
        // Safety: This default implementation is only used by scalar strategies which use QueryStack.
        // SIMD strategies override this entire method.
        let stack_ref = unsafe {
            &mut *(stack as *mut Self::Stack<O>
                as *mut crate::kd_tree::query_stack::QueryStack<O, Self>)
        };
        tree.backtracking_query_with_stack_impl::<QC, O, D>(query_ctx, stack_ref, process_leaf);
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::Index;

    #[test]
    fn test_u16() {
        assert_eq!(<u16 as Index>::max(), u16::MAX);
        assert_eq!(<u16 as Index>::min(), 0u16);
        assert_eq!(<u16 as Index>::leaf_offset(), 32_767u16);
        assert_eq!(256u16.ilog2(), 8u32);
        assert_eq!(u16::capacity_with_bucket_size(32), 1_048_576);
    }

    #[test]
    fn test_u32() {
        assert_eq!(<u32 as Index>::max(), u32::MAX);
        assert_eq!(<u32 as Index>::min(), 0u32);
        assert_eq!(<u32 as Index>::leaf_offset(), 2_147_483_647);
        assert_eq!(256u32.ilog2(), 8u32);

        #[cfg(target_pointer_width = "64")]
        assert_eq!(u32::capacity_with_bucket_size(32), 68_719_476_736);

        #[cfg(target_pointer_width = "32")]
        assert_eq!(u32::capacity_with_bucket_size(32), u32::MAX);
    }
    #[test]
    fn test_u32_simulate_32bit_target_pointer() {
        // TODO: replace this with wasm-bindgen-tests at some point
        let bucket_size: u32 = 32;
        let capacity_with_bucket_size =
            (u32::MAX - u32::MAX.overflowing_shr(1).0).saturating_mul(bucket_size);
        assert_eq!(capacity_with_bucket_size, u32::MAX);
    }
}
