//! Donnelly SIMD + Prefetch Stem Strategy

use crate::stem_strategies::donnelly_2_blockmarker_simd::avx2_impl::{
    compare_block3_f32_avx2, compare_block3_f64_avx2, compare_block4_f32_avx2,
};
use crate::stem_strategies::donnelly_core::DonnellyCore;
use crate::stem_strategies::{Block3, Block4, BlockSizeMarker};
use crate::traits_unified_2::AxisUnified;
use crate::StemStrategy;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Block3 interval lower bounds encoded as u64 literal.
/// Each 8-bit segment contains the lower bound pivot offset for a child (255 = -∞).
/// Lower bounds: [255, 3, 1, 4, 0, 5, 2, 6] for children 0-7
const CHILD_LOWER_BOUNDS_BLOCK3: u64 = 0x06_02_05_00_04_01_03_FF;

/// Block3 interval upper bounds encoded as u64 literal.
/// Each 8-bit segment contains the upper bound pivot offset for a child (255 = +∞).
/// Upper bounds: [3, 1, 4, 0, 5, 2, 6, 255] for children 0-7
const CHILD_UPPER_BOUNDS_BLOCK3: u64 = 0xFF_06_02_05_00_04_01_03;

/// Returns the interval bounds [lower, upper) for a Block3 child in a given dimension.
///
/// Block3 has 8 children arranged in a triangular layout, where all levels in the block
/// split on the same dimension. Each child occupies an interval [lower, upper) in that dimension.
///
/// Returns (lower_pivot_offset, upper_pivot_offset) where:
/// - Offset refers to the pivot within the block (0-6 are actual pivots, 7 is padding)
/// - 255 represents ±∞ (child 0 extends to -∞, child 7 extends to +∞)
///
/// # Example
/// ```text
/// Block structure:
///           pivot[0]
///      pivot[1]    pivot[2]
///   p[3] p[4]   p[5] p[6]
/// ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7
///
/// Child 0: [-∞, pivot[3]) → (255, 3)
/// Child 1: [pivot[3], pivot[1]) → (3, 1)
/// Child 2: [pivot[1], pivot[4]) → (1, 4)
/// ...etc
/// ```
#[inline(always)]
const fn child_interval_bounds_block3(child_idx: usize) -> (u8, u8) {
    let lower = ((CHILD_LOWER_BOUNDS_BLOCK3 >> (child_idx * 8)) & 0xFF) as u8;
    let upper = ((CHILD_UPPER_BOUNDS_BLOCK3 >> (child_idx * 8)) & 0xFF) as u8;
    (lower, upper)
}

/// Computes absolute distance from a query point to an interval [lower, upper).
///
/// Branchless implementation for performance.
///
/// TODO: This is hardcoded for SquaredEuclidean metric. Generalize to Manhattan and other
///       metrics once we confirm this interval-based approach works correctly.
///
/// Returns:
/// - If query < lower: lower - query
/// - If query >= upper: query - upper
/// - If lower <= query < upper: 0 (query is inside the interval)
#[inline(always)]
fn interval_distance_1d(query: f64, lower: f64, upper: f64) -> f64 {
    let below_dist = (lower - query).max(0.0);
    let above_dist = (query - upper).max(0.0);
    below_dist + above_dist
}

/// Donnelly SIMD Strategy
///
/// Donnelly ordering, block-at-once evaluation.
/// Switches dimension once per block rather than every level.
/// Delegates to DonnellyCore for traversal at construction time,
/// but has custom query-time traversal methods for block-at-once traversal.
///
/// - BS: Block size, i.e. minor tri height.
/// - CL: Cache line width in bytes (64, most of the time. Can be 128 for Apple M2+)
/// - VB: Value width in bytes (e.g. 4 for f32, 8 for f64)
/// - K: Dimensionality
#[derive(Copy, Clone, Debug)]
pub struct DonnellyMarkerSimd<BS: BlockSizeMarker, const CL: u32, const VB: u32, const K: usize> {
    core: DonnellyCore<CL, VB, K>,
    _marker: PhantomData<BS>,
}

// x86_64 3-level (eg with f64)
// (All x86_64 processors have a cache line size of 64 bytes)
#[cfg(target_arch = "x86_64")]
impl<const VB: u32, const K: usize> StemStrategy for DonnellyMarkerSimd<Block3, 64, VB, K> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;

    #[cfg(feature = "simd")]
    type StackContext<A> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<A, Self>;
    #[cfg(feature = "simd")]
    type Stack<A> = crate::kd_tree::query_stack_simd::SimdQueryStack<A, Self>;

    #[cfg(not(feature = "simd"))]
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self>;
    #[cfg(not(feature = "simd"))]
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        debug_assert!(64 >= VB); // item wider than cache line would break layout

        Self {
            core: DonnellyCore::new(stems_ptr),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.core.stem_idx()
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.core.leaf_idx()
    }

    #[inline(always)]
    fn dim(&self) -> usize {
        self.core.dim()
    }

    #[inline(always)]
    fn construction_dim(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse(&mut self, is_right: bool) {
        self.core.traverse(is_right)
    }

    #[inline(always)]
    fn branch(&mut self) -> Self {
        Self {
            core: self.core.branch(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn child_indices(&self) -> (usize, usize) {
        self.core.child_indices()
    }

    fn get_leaf_idx<A: AxisUnified, const K2: usize>(
        stems: &[A],
        query: &[A; K2],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        // Note: a precondition of this approach is that the tree must have been constructed
        // so that the stem height is an exact multiple of the block size, and that the stems
        // all terminate on that same max level. This invariant is guaranteed at construction
        // time.
        while strat.level() <= max_stem_level {
            let dim = strat.dim();
            let query_val = unsafe { *query.get_unchecked(dim) };
            let block_base_idx = strat.stem_idx();

            let child_idx = compare_block3(stems, query_val, block_base_idx);

            strat
                .core
                .traverse_block(child_idx, Self::BLOCK_SIZE as u32);
        }

        strat.leaf_idx()
    }

    #[inline(always)]
    fn backtracking_traverse_step<A, O, D, const K2: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K2],
        query_wide: &[O; K2],
        off: &mut [O; K2],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: AxisUnified<Coord = A> + CompareBlock3,
        O: AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
    {
        // Check if we've finished traversing (reached or passed max_stem_level)
        // For block strategies, we traverse multiple levels at once
        if self.level() > max_stem_level {
            return false;
        }

        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };
        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();

        // SIMD comparison to get child index
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let child_idx = CompareBlock3::compare_block3_impl(stems_ptr, query_val, block_base_idx);
        tracing::trace!("child_idx = {}", child_idx);

        // SIMD distance check to get backtrack mask
        // Dispatch based on axis type size, similar to compare_block3_impl
        let backtrack_mask = if std::mem::size_of::<A>() == 8 {
            // f64 path
            let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f64: f64 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f64: f64 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f64: f64 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                D::simd_backtrack_check_block3_f64_avx512(
                    unsafe { std::mem::transmute_copy(&query_wide_f64) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f64) },
                    unsafe { std::mem::transmute_copy(&rd_f64) },
                    unsafe { std::mem::transmute_copy(&best_dist_f64) },
                )
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe {
                    D::simd_backtrack_check_block3_interval_f64_avx2(
                        std::mem::transmute_copy(&query_wide_f64),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f64),
                        std::mem::transmute_copy(&rd_f64),
                        std::mem::transmute_copy(&best_dist_f64),
                    )
                }
            }
        } else if std::mem::size_of::<A>() == 4 {
            // f32 path
            let query_wide_f32: f32 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f32: f32 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f32: f32 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f32: f32 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe {
                    D::simd_backtrack_check_block3_f32_avx512(
                        std::mem::transmute_copy(&query_wide_f32),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f32),
                        std::mem::transmute_copy(&rd_f32),
                        std::mem::transmute_copy(&best_dist_f32),
                    )
                }
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe {
                    D::simd_backtrack_check_block3_f32_avx2(
                        std::mem::transmute_copy(&query_wide_f32),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f32),
                        std::mem::transmute_copy(&rd_f32),
                        std::mem::transmute_copy(&best_dist_f32),
                    )
                }
            }
        } else {
            panic!("Unsupported axis type size");
        };

        // Remove the path we're taking from backtrack candidates
        let child_idx_mask = 1 << child_idx;
        let backtrack_mask = backtrack_mask & !child_idx_mask;

        // Log all 7 pivots for debugging
        let pivots: Vec<A> = (0..7)
            .map(|i| unsafe { *stems.get_unchecked(block_base_idx + i) })
            .collect();

        tracing::trace!(
            child_idx,
            stem_idx = self.core.stem_idx(),
            dim = *dim,
            backtrack_mask_before = backtrack_mask | child_idx_mask,
            backtrack_mask_after = backtrack_mask,
            taking_path = format!("child {}", child_idx),
            pivots = ?pivots,
            "Block3: backtrack mask"
        );

        // If any siblings need exploration, push them as a Block entry
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            // TODO: this is too slow. Need to:
            //   * Just store a clone of self, rather than creating all the siblings and traversing them
            //     all here. We should only create the sibling and traverse it after we perform the SIMD prune.
            //   * In cases where D == O, widen is a no-op. In this case we can use SIMD ops to calc new_off
            //     and rd_values in parallel for all siblings.

            // Calculate all sibling contexts and their rd_far values
            let mut siblings = [self.clone(); 8];
            let mut rd_values = [O::zero(); 8];
            let mut new_off_values = [O::zero(); 8];

            for sibling_idx in 0..8 {
                // Create context for this sibling
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                // Calculate rd_far for this sibling using interval distance
                let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx);

                // Get lower bound (-∞ if offset is 255)
                let lower = if lower_offset == 255 {
                    A::min_value()
                } else {
                    unsafe { *stems.get_unchecked(block_base_idx + lower_offset as usize) }
                };

                // Get upper bound (+∞ if offset is 255)
                let upper = if upper_offset == 255 {
                    A::max_value()
                } else {
                    unsafe { *stems.get_unchecked(block_base_idx + upper_offset as usize) }
                };

                // Compute interval distance
                let query_val = unsafe { *query.get_unchecked(*dim) };

                // TODO: this only works for f64! need a refactor to work for any AxisUnified
                let query_wide_f64: f64 =
                    unsafe { std::mem::transmute_copy(&D::widen_coord(query_val)) };
                let lower_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(lower)) };
                let upper_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(upper)) };
                let interval_dist = interval_distance_1d(query_wide_f64, lower_f64, upper_f64);
                let new_off: O = unsafe { std::mem::transmute_copy(&interval_dist) };

                new_off_values[sibling_idx] = new_off;

                // rd_far = rd + dist1(new_off, old_off) = rd + (new_off - old_off)²
                // Using the standard backtracking formula from traits.rs:394
                let delta = D::dist1(new_off, old_off_val);
                let rd_far = O::saturating_add(rd, delta);
                rd_values[sibling_idx] = rd_far;

                tracing::debug!(
                    sibling_idx,
                    stem_idx = self.core.stem_idx(),
                    dim = *dim,
                    lower_offset,
                    upper_offset,
                    ?lower,
                    ?upper,
                    lower_f64,
                    upper_f64,
                    query_val_f64 = query_wide_f64,
                    interval_dist,
                    ?new_off,
                    ?old_off_val,
                    ?rd,
                    ?delta,
                    ?rd_far,
                    ?best_dist,
                    passes_threshold = rd_far <= best_dist,
                    in_backtrack_mask = (backtrack_mask & (1 << sibling_idx)) != 0,
                    "SIMD Block3: sibling interval calc"
                );
            }

            // Push single Block entry containing all siblings
            stack.push(SimdQueryStackContext::Block {
                siblings,
                rd_values,
                new_off_values,
                sibling_mask: backtrack_mask,
                dim: dim_val,
                old_off: old_off_val,
            });
        }

        // Update off[dim] with the new_off for the path we're taking
        // Use the same interval distance formula as for siblings for consistency
        let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx as usize);

        let lower = if lower_offset == 255 {
            A::min_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + lower_offset as usize) }
        };

        let upper = if upper_offset == 255 {
            A::max_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + upper_offset as usize) }
        };

        // Compute interval distance for the chosen child (should be 0 or very small)
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(query_val)) };
        let lower_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(lower)) };
        let upper_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(upper)) };

        let interval_dist = interval_distance_1d(query_wide_f64, lower_f64, upper_f64);
        let new_off: O = unsafe { std::mem::transmute_copy(&interval_dist) };
        unsafe { *off.get_unchecked_mut(dim_val) = new_off };

        // Traverse to the chosen child
        self.core.traverse_block(child_idx, Self::BLOCK_SIZE as u32);

        // Update dim
        *dim = self.dim();

        true
    }

    #[cfg(feature = "simd")]
    fn backtracking_query_with_stack<A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &crate::kd_tree::KdTree<A, T, Self, LS, K2, B>,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(&crate::kd_tree::leaf_view::LeafView<A, T, K2, B>, &mut QC),
    ) where
        Self: Sized,
        A: crate::traits_unified_2::AxisUnified<Coord = A> + CompareBlock3,
        T: crate::traits_unified_2::Basics + Copy + Default + PartialOrd + PartialEq,
        O: crate::traits_unified_2::AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
        QC: crate::kd_tree::traits::QueryContext<A, O, K2>,
        LS: crate::traits_unified_2::LeafStrategy<A, T, Self, K2, B>,
    {
        // Delegate to KdTree's SIMD implementation which has access to private fields
        tree.backtracking_query_with_simd_stack_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

// x86_64 4-level (eg with f32)
// (All x86_64 processors have a cache line size of 64 bytes)
#[cfg(target_arch = "x86_64")]
impl<const VB: u32, const K: usize> StemStrategy for DonnellyMarkerSimd<Block4, 64, VB, K> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 4;

    #[cfg(feature = "simd")]
    type StackContext<A> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<A, Self>;
    #[cfg(feature = "simd")]
    type Stack<A> = crate::kd_tree::query_stack_simd::SimdQueryStack<A, Self>;

    #[cfg(not(feature = "simd"))]
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self>;
    #[cfg(not(feature = "simd"))]
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        debug_assert!(64 >= VB); // item wider than cache line would break layout

        Self {
            core: DonnellyCore::new(stems_ptr),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.core.stem_idx()
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.core.leaf_idx()
    }

    #[inline(always)]
    fn dim(&self) -> usize {
        self.core.dim()
    }

    #[inline(always)]
    fn construction_dim(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse(&mut self, is_right: bool) {
        self.core.traverse(is_right)
    }

    #[inline(always)]
    fn branch(&mut self) -> Self {
        Self {
            core: self.core.branch(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn child_indices(&self) -> (usize, usize) {
        self.core.child_indices()
    }

    fn get_leaf_idx<A: AxisUnified, const K2: usize>(
        stems: &[A],
        query: &[A; K2],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        // Note: a precondition of this approach is that the tree must have been constructed
        // so that the stem height is an exact multiple of the block size, and that the stems
        // all terminate on that same max level. This invariant is guaranteed at construction
        // time.
        while strat.level() <= max_stem_level {
            let dim = strat.dim();
            let query_val = unsafe { *query.get_unchecked(dim) };
            let block_base_idx = strat.stem_idx();

            let child_idx = compare_block4(stems, query_val, block_base_idx);

            strat
                .core
                .traverse_block(child_idx, Self::BLOCK_SIZE as u32);
        }

        strat.leaf_idx()
    }

    #[inline(always)]
    fn backtracking_traverse_step<A, O, D, const K2: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K2],
        query_wide: &[O; K2],
        off: &mut [O; K2],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: AxisUnified<Coord = A> + CompareBlock4,
        O: AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
    {
        // Check if we've finished traversing (reached or passed max_stem_level)
        // For block strategies, we traverse multiple levels at once
        if self.level() > max_stem_level {
            return false;
        }

        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };
        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();

        // SIMD comparison to get child index
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let child_idx = CompareBlock4::compare_block4_impl(stems_ptr, query_val, block_base_idx);

        // SIMD distance check to get backtrack mask
        let backtrack_mask = if std::mem::size_of::<A>() == 8 {
            // f64 path
            let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f64: f64 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f64: f64 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f64: f64 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe {
                    D::simd_backtrack_check_block4_f64_avx512(
                        std::mem::transmute_copy(&query_wide_f64),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f64),
                        std::mem::transmute_copy(&rd_f64),
                        std::mem::transmute_copy(&best_dist_f64),
                    )
                }
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe {
                    D::simd_backtrack_check_block4_f64_avx2(
                        std::mem::transmute_copy(&query_wide_f64),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f64),
                        std::mem::transmute_copy(&rd_f64),
                        std::mem::transmute_copy(&best_dist_f64),
                    )
                }
            }
        } else if std::mem::size_of::<A>() == 4 {
            // f32 path
            let query_wide_f32: f32 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f32: f32 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f32: f32 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f32: f32 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe {
                    D::simd_backtrack_check_block4_f32_avx512(
                        std::mem::transmute_copy(&query_wide_f32),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f32),
                        std::mem::transmute_copy(&rd_f32),
                        std::mem::transmute_copy(&best_dist_f32),
                    )
                }
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe {
                    D::simd_backtrack_check_block4_f32_avx2(
                        std::mem::transmute_copy(&query_wide_f32),
                        stems_ptr.as_ptr(),
                        block_base_idx,
                        std::mem::transmute_copy(&old_off_f32),
                        std::mem::transmute_copy(&rd_f32),
                        std::mem::transmute_copy(&best_dist_f32),
                    )
                }
            }
        } else {
            panic!("Unsupported axis type size");
        };

        // Remove the path we're taking from backtrack candidates
        let backtrack_mask = backtrack_mask & !(1 << child_idx);

        // If any siblings need exploration, push them as Block entries
        // Block4 has 16 siblings, so push two blocks of 8
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            // Calculate all sibling contexts and their rd_far values
            let mut siblings = [self.clone(); 16];
            let mut rd_values = [O::zero(); 16];
            let mut new_off_values = [O::zero(); 16];

            for sibling_idx in 0..16 {
                // Create context for this sibling
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                // Calculate rd_far for this sibling
                let pivot_idx = block_base_idx + sibling_idx;
                let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
                let pivot_wide = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                new_off_values[sibling_idx] = new_off;
                rd_values[sibling_idx] = O::saturating_add(rd, D::dist1(new_off, old_off_val));
            }

            // Push high block (siblings 8-15) first, since stack is LIFO
            let high_mask = (backtrack_mask >> 8) as u8;
            if high_mask != 0 {
                let mut high_siblings = [self.clone(); 8];
                let mut high_rd_values = [O::zero(); 8];
                let mut high_new_off_values = [O::zero(); 8];
                for i in 0..8 {
                    high_siblings[i] = siblings[i + 8].clone();
                    high_rd_values[i] = rd_values[i + 8];
                    high_new_off_values[i] = new_off_values[i + 8];
                }
                stack.push(SimdQueryStackContext::Block {
                    siblings: high_siblings,
                    rd_values: high_rd_values,
                    new_off_values: high_new_off_values,
                    sibling_mask: high_mask,
                    dim: dim_val,
                    old_off: old_off_val,
                });
            }

            // Push low block (siblings 0-7)
            let low_mask = backtrack_mask as u8;
            if low_mask != 0 {
                let mut low_siblings = [self.clone(); 8];
                let mut low_rd_values = [O::zero(); 8];
                let mut low_new_off_values = [O::zero(); 8];
                for i in 0..8 {
                    low_siblings[i] = siblings[i].clone();
                    low_rd_values[i] = rd_values[i];
                    low_new_off_values[i] = new_off_values[i];
                }
                stack.push(SimdQueryStackContext::Block {
                    siblings: low_siblings,
                    rd_values: low_rd_values,
                    new_off_values: low_new_off_values,
                    sibling_mask: low_mask,
                    dim: dim_val,
                    old_off: old_off_val,
                });
            }
        }

        // Update off[dim] with the new_off for the path we're taking
        let pivot_idx = block_base_idx + child_idx as usize;
        let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
        let pivot_wide = D::widen_coord(pivot);
        let new_off = O::saturating_dist(query_wide_val, pivot_wide);
        unsafe { *off.get_unchecked_mut(dim_val) = new_off };

        // Traverse to the chosen child
        self.core.traverse_block(child_idx, Self::BLOCK_SIZE as u32);

        // Update dim
        *dim = self.dim();

        true
    }

    #[cfg(feature = "simd")]
    fn backtracking_query_with_stack<A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &crate::kd_tree::KdTree<A, T, Self, LS, K2, B>,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(&crate::kd_tree::leaf_view::LeafView<A, T, K2, B>, &mut QC),
    ) where
        Self: Sized,
        A: crate::traits_unified_2::AxisUnified<Coord = A> + CompareBlock4,
        T: crate::traits_unified_2::Basics + Copy + Default + PartialOrd + PartialEq,
        O: crate::traits_unified_2::AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
        QC: crate::kd_tree::traits::QueryContext<A, O, K2>,
        LS: crate::traits_unified_2::LeafStrategy<A, T, Self, K2, B>,
    {
        // Delegate to KdTree's SIMD implementation which has access to private fields
        tree.backtracking_query_with_simd_stack_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

/// Perform all comparisons in a 3-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
fn compare_block3<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: AxisUnified + CompareBlock3,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

    // Type-based dispatch via trait specialization pattern
    CompareBlock3::compare_block3_impl(stems_ptr, query_val, block_base_idx)
}

/// Trait for SIMD block comparison support.
///
/// This trait is automatically implemented for all `AxisUnified` types.
/// Specialized SIMD implementations exist for:
/// - `f32` - uses AVX2 or AVX-512
/// - `f64` - uses AVX2 or AVX-512
///
/// Users should not implement this trait directly. If you try to use `DonnellyMarkerSimd`
/// with an unsupported type, you'll get a runtime panic.
pub trait CompareBlock3: AxisUnified {
    #[doc(hidden)]
    fn compare_block3_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8;
}

// Blanket implementation that dispatches based on type
impl<T: AxisUnified> CompareBlock3 for T {
    #[inline(always)]
    fn compare_block3_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        // Type-based dispatch using size_of and transmute
        // This is monomorphized away at compile time
        if std::mem::size_of::<Self>() == 8 {
            // f64 path
            let query_f64: f64 = unsafe { std::mem::transmute_copy(&query_val) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe { compare_block3_f64_avx512(stems_ptr, block_base_idx, query_f64) }
            }

            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe { compare_block3_f64_avx2(stems_ptr, block_base_idx, query_f64) }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                panic!("DonnellyMarkerSimd requires x86_64 with AVX2")
            }
        } else if std::mem::size_of::<Self>() == 4 {
            // f32 path
            let query_f32: f32 = unsafe { std::mem::transmute_copy(&query_val) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe { compare_block3_f32_avx512(stems_ptr, block_base_idx, query_f32) }
            }

            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe { compare_block3_f32_avx2(stems_ptr, block_base_idx, query_f32) }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                panic!("DonnellyMarkerSimd requires x86_64 with AVX2")
            }
        } else {
            panic!(
                "DonnellyMarkerSimd only supports f32 (4 bytes) and f64 (8 bytes) axis types. \
                    Type {} with size {} is not supported.",
                std::any::type_name::<Self>(),
                std::mem::size_of::<Self>()
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::ptr::NonNull;

    #[target_feature(enable = "avx2,popcnt")]
    #[inline(always)]
    pub unsafe fn compare_block3_f64_avx2(
        stems_ptr: NonNull<u8>,
        cache_line_base: usize,
        query_val: f64,
    ) -> u8 {
        use std::arch::x86_64::*;

        let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;

        let pivots_low = _mm256_loadu_pd(ptr);
        let pivots_high = _mm256_loadu_pd(ptr.add(4));
        let query_vec = _mm256_set1_pd(query_val);

        let cmp_low = _mm256_cmp_pd(query_vec, pivots_low, _CMP_GE_OQ);
        let cmp_high = _mm256_cmp_pd(query_vec, pivots_high, _CMP_GE_OQ);

        let mask_low = _mm256_movemask_pd(cmp_low) as u32;
        let mask_high = _mm256_movemask_pd(cmp_high) as u32;
        let mask = mask_low | (mask_high << 4);

        _popcnt32(mask as i32) as u8
    }

    #[target_feature(enable = "avx2,popcnt")]
    #[inline(always)]
    pub unsafe fn compare_block3_f32_avx2(
        stems_ptr: NonNull<u8>,
        cache_line_base: usize,
        query_val: f32,
    ) -> u8 {
        use std::arch::x86_64::*;

        let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;

        let pivots = _mm256_loadu_ps(ptr);
        let query_vec = _mm256_set1_ps(query_val);

        let cmp = _mm256_cmp_ps(query_vec, pivots, _CMP_GE_OQ);

        let mask = _mm256_movemask_ps(cmp) as u32;

        _popcnt32(mask as i32) as u8
    }

    #[target_feature(enable = "avx2,popcnt")]
    #[inline(always)]
    pub unsafe fn compare_block4_f32_avx2(
        stems_ptr: NonNull<u8>,
        cache_line_base: usize,
        query_val: f32,
    ) -> u8 {
        use std::arch::x86_64::*;

        let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;

        let pivots_low = _mm256_loadu_ps(ptr);
        let pivots_high = _mm256_loadu_ps(ptr.add(8));
        let query_vec = _mm256_set1_ps(query_val);

        let cmp_low = _mm256_cmp_ps(query_vec, pivots_low, _CMP_GE_OQ);
        let cmp_high = _mm256_cmp_ps(query_vec, pivots_high, _CMP_GE_OQ);

        let mask_low = _mm256_movemask_ps(cmp_low) as u32;
        let mask_high = _mm256_movemask_ps(cmp_high) as u32;
        let mask = mask_low | (mask_high << 8);

        _popcnt32(mask as i32) as u8
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512_impl {
    #[target_feature(enable = "avx512f,avx512vl,popcnt")]
    #[inline(always)]
    pub unsafe fn compare_block3_f64_avx512(
        stems_ptr: NonNull<u8>,
        cache_line_base: usize,
        query_val: f64,
    ) -> u8 {
        use std::arch::x86_64::*;

        let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;
        let pivots = _mm512_loadu_pd(ptr);
        let query_vec = _mm512_set1_pd(query_val);

        let mask = _mm512_cmp_pd_mask(query_vec, pivots, _CMP_GE_OQ);
        _popcnt32(mask as i32) as u8
    }

    #[target_feature(enable = "avx512f,avx512vl,popcnt")]
    #[inline(always)]
    pub unsafe fn compare_block4_f32_avx512(
        stems_ptr: NonNull<u8>,
        cache_line_base: usize,
        query_val: f32,
    ) -> u8 {
        use std::arch::x86_64::*;

        let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f32;
        let pivots = _mm512_loadu_ps(ptr);
        let query_vec = _mm512_set1_ps(query_val);

        let mask = _mm512_cmp_ps_mask(query_vec, pivots, _CMP_GE_OQ);
        _popcnt32(mask as i32) as u8
    }
}

/// Perform all comparisons in a 4-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
fn compare_block4<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: AxisUnified + CompareBlock4,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

    // Type-based dispatch via trait specialization pattern
    CompareBlock4::compare_block4_impl(stems_ptr, query_val, block_base_idx)
}

/// Trait for SIMD 4-level block comparison support.
///
/// This trait is automatically implemented for all `AxisUnified` types.
/// Specialized SIMD implementations exist for:
/// - `f32` - uses AVX2 or AVX-512
/// - `f64` - uses AVX2 or AVX-512
///
/// Users should not implement this trait directly. If you try to use `DonnellyMarkerSimd`
/// with an unsupported type, you'll get a runtime panic.
pub trait CompareBlock4: AxisUnified {
    #[doc(hidden)]
    fn compare_block4_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8;
}

// Blanket implementation that dispatches based on type
impl<T: AxisUnified> CompareBlock4 for T {
    #[inline(always)]
    fn compare_block4_impl(stems_ptr: NonNull<u8>, query_val: Self, block_base_idx: usize) -> u8 {
        // Type-based dispatch using size_of and transmute
        // This is monomorphized away at compile time
        if std::mem::size_of::<Self>() == 4 {
            // f32 path
            let query_f32: f32 = unsafe { std::mem::transmute_copy(&query_val) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                unsafe { compare_block4_f32_avx512(stems_ptr, block_base_idx, query_f32) }
            }

            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                unsafe { compare_block4_f32_avx2(stems_ptr, block_base_idx, query_f32) }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                panic!("DonnellyMarkerSimd requires x86_64 with AVX2")
            }
        } else {
            panic!(
                "DonnellyMarkerSimd only supports f32 (4 bytes) axis types. \
                    Type {} with size {} is not supported.",
                std::any::type_name::<Self>(),
                std::mem::size_of::<Self>()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_child_interval_bounds_block3() {
        // Verify all 8 children have correct interval bounds
        // Expected bounds based on triangular structure:
        //           pivot[0]
        //      pivot[1]    pivot[2]
        //   p[3] p[4]   p[5] p[6]
        // ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7

        assert_eq!(child_interval_bounds_block3(0), (255, 3)); // [-∞, pivot[3])
        assert_eq!(child_interval_bounds_block3(1), (3, 1)); // [pivot[3], pivot[1])
        assert_eq!(child_interval_bounds_block3(2), (1, 4)); // [pivot[1], pivot[4])
        assert_eq!(child_interval_bounds_block3(3), (4, 0)); // [pivot[4], pivot[0])
        assert_eq!(child_interval_bounds_block3(4), (0, 5)); // [pivot[0], pivot[5])
        assert_eq!(child_interval_bounds_block3(5), (5, 2)); // [pivot[5], pivot[2])
        assert_eq!(child_interval_bounds_block3(6), (2, 6)); // [pivot[2], pivot[6])
        assert_eq!(child_interval_bounds_block3(7), (6, 255)); // [pivot[6], +∞)
    }

    #[test]
    fn test_interval_distance_1d_inside() {
        // Query inside interval should return 0
        assert_eq!(interval_distance_1d(5.0, 3.0, 7.0), 0.0);
        assert_eq!(interval_distance_1d(3.0, 3.0, 7.0), 0.0); // At lower bound
        assert_eq!(interval_distance_1d(6.999, 3.0, 7.0), 0.0); // Just below upper bound
    }

    #[test]
    fn test_interval_distance_1d_below() {
        // Query below lower bound
        assert_eq!(interval_distance_1d(2.0, 5.0, 10.0), 3.0); // |5 - 2| = 3
        assert_eq!(interval_distance_1d(0.0, 3.0, 10.0), 3.0); // |3 - 0| = 3
        assert_eq!(interval_distance_1d(-1.0, 1.0, 10.0), 2.0); // |1 - (-1)| = 2
    }

    #[test]
    fn test_interval_distance_1d_above() {
        // Query above upper bound
        assert_eq!(interval_distance_1d(12.0, 5.0, 10.0), 2.0); // |12 - 10| = 2
        assert_eq!(interval_distance_1d(10.0, 5.0, 10.0), 0.0); // Exactly at upper bound (excluded)
        assert_eq!(interval_distance_1d(15.0, 5.0, 10.0), 5.0); // |15 - 10| = 5
    }

    #[test]
    fn test_interval_distance_1d_edge_cases() {
        // Test with infinity bounds
        assert_eq!(interval_distance_1d(-100.0, f64::NEG_INFINITY, 5.0), 0.0); // Below but no lower bound
        assert_eq!(interval_distance_1d(100.0, 5.0, f64::INFINITY), 0.0); // Above but no upper bound
        assert_eq!(
            interval_distance_1d(0.0, f64::NEG_INFINITY, f64::INFINITY),
            0.0
        ); // Unbounded
    }

    #[test]
    fn test_interval_distance_1d_branchless_correctness() {
        // Test that branchless impl gives same results as branching version
        for query in [-10.0, -1.0, 0.0, 1.0, 5.0, 7.5, 10.0, 15.0, 100.0] {
            let lower = 0.0;
            let upper = 10.0;

            let result = interval_distance_1d(query, lower, upper);

            // Compute expected with explicit branches
            let expected = if query < lower {
                lower - query
            } else if query >= upper {
                query - upper
            } else {
                0.0
            };

            assert_eq!(result, expected, "Failed for query={}", query);
        }
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn test_simd_backtrack_vs_scalar() {
    use crate::traits_unified_2::DistanceMetricUnified;
    use crate::traits_unified_2::SquaredEuclidean;

    // Create test pivots: [pivot0, pivot1, ..., pivot6, +∞]
    let pivots = [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY];

    let query = 0.25;
    let old_off = 0.0;
    let rd = 0.0;
    let best_dist = f64::INFINITY;

    // Compute scalar version for each child
    let mut scalar_results = [false; 8];
    for child_idx in 0..8 {
        let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

        let lower = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };

        let upper = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };

        let interval_dist = interval_distance_1d(query, lower, upper);
        // rd_far = rd + (interval_dist - old_off)²
        let delta = (interval_dist - old_off) * (interval_dist - old_off);
        let rd_far = rd + delta;

        scalar_results[child_idx] = rd_far <= best_dist;
    }

    // Compute SIMD version
    let pivots_ptr = pivots.as_ptr() as *const u8;
    // Replace the test with a simpler version
    let simd_mask = unsafe {
        <SquaredEuclidean<f64> as DistanceMetricUnified<f64, 3>>::simd_backtrack_check_block3_interval_f64_avx2(
        query, pivots_ptr, 0, old_off, rd, best_dist
    )
    };

    // Compare
    for child_idx in 0..8 {
        let scalar_pass = scalar_results[child_idx];
        let simd_pass = (simd_mask & (1 << child_idx)) != 0;
        assert_eq!(
            scalar_pass,
            simd_pass,
            "Mismatch for child {}: scalar={}, simd={}, query={}, lower={:?}, upper={:?}",
            child_idx,
            scalar_pass,
            simd_pass,
            query,
            if child_interval_bounds_block3(child_idx).0 == 255 {
                f64::NEG_INFINITY
            } else {
                pivots[child_interval_bounds_block3(child_idx).0 as usize]
            },
            if child_interval_bounds_block3(child_idx).1 == 255 {
                f64::INFINITY
            } else {
                pivots[child_interval_bounds_block3(child_idx).1 as usize]
            },
        );
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn debug_query_12_interval_distances() {
    // This test manually computes what should happen for query #12
    // Query point: [0.8947785353168005, 0.678720516865904, 0.6048091301041568]
    // We're interested in the second block pop (dim=0, old_off=0)

    // From the log, the pivots that would be in a block starting at some block_base_idx
    // Let's trace through what the interval distances should be

    // rd_values from log: [0.5638408493966387, 0.3959802548889544, 0.25295146890380843,
    //                       0.14310568836537063, 0.06606798320124753, 0.0170270603870678,
    //                       0.00021225610203875987, 0.0]

    // For each child, let's print what the intervals should be
    println!("Query value in dim 0: 0.8947785353168005");
    println!("\nChild interval bounds and expected distances:");

    for child_idx in 0..8 {
        let (lower_off, upper_off) = child_interval_bounds_block3(child_idx);
        println!(
            "Child {}: lower_offset={}, upper_offset={}",
            child_idx, lower_off, upper_off
        );
    }

    // The issue is: why does child 6 pass but not child 4?
    // best_dist at that point should be 0.0036181109111460682
    println!("\nbest_dist would be: 0.0036181109111460682");
    println!(
        "Child 4 rd_value: 0.06606798320124753 > best_dist? {}",
        0.06606798320124753 > 0.0036181109111460682
    );
    println!(
        "Child 6 rd_value: 0.00021225610203875987 > best_dist? {}",
        0.00021225610203875987 > 0.0036181109111460682
    );

    // So child 6 should indeed survive and child 4 should be pruned based on rd_values
    // But non-SIMD found the answer in leaf 52 (child 4's leaf)
    // This suggests either:
    // 1. The interval distance calculation is wrong
    // 2. The non-SIMD uses different logic
    // 3. There's something about the tree structure we're missing
}
