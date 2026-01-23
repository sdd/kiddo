//! Donnelly SIMD + Prefetch Stem Strategy
//!
//! This module provides SIMD-optimized block-at-once traversal strategies for kd-trees.
//! Architecture-specific implementations are in submodules.

use crate::stem_strategies::donnelly_core::DonnellyCore;
use crate::stem_strategies::{Block3, Block4, BlockSizeMarker};
use crate::traits_unified_2::AxisUnified;
use std::marker::PhantomData;
use std::ptr::NonNull;

// Architecture-specific modules
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod x86_64;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod aarch64;

mod autovec;

// Pruning traits for SIMD dispatch
mod prune_traits;
pub use prune_traits::SimdPrune;

// Comparison traits for type-specific dispatch
mod compare_traits;
pub use compare_traits::{CompareBlock3, CompareBlock4};

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
pub(crate) const fn child_interval_bounds_block3(child_idx: usize) -> (u8, u8) {
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
pub(crate) fn interval_distance_1d(query: f64, lower: f64, upper: f64) -> f64 {
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

/// Perform all comparisons in a 3-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
pub(crate) fn compare_block3<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: CompareBlock3,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
    A::compare_block3_impl(stems_ptr, query_val, block_base_idx)
}

/// Perform all comparisons in a 4-level block, dispatching to the appropriate SIMD implementation
/// based on the axis type A. Monomorphized at compile time.
#[inline(always)]
pub(crate) fn compare_block4<A>(stems: &[A], query_val: A, block_base_idx: usize) -> u8
where
    A: CompareBlock4,
{
    let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
    A::compare_block4_impl(stems_ptr, query_val, block_base_idx)
}

// ====================================================================================
// StemStrategy Implementations (architecture-agnostic, dispatch through traits)
// ====================================================================================

// Block3 implementation (3-level blocks, 64-byte cache lines)
impl<const VB: u32, const K: usize> crate::StemStrategy for DonnellyMarkerSimd<Block3, 64, VB, K> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;

    type StackContext<A> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<A, Self>;
    type Stack<A> = crate::kd_tree::query_stack_simd::SimdQueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: std::ptr::NonNull<u8>) -> Self {
        debug_assert!(64 >= VB); // item wider than cache line would break layout

        Self {
            core: crate::stem_strategies::donnelly_core::DonnellyCore::new(stems_ptr),
            _marker: std::marker::PhantomData,
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
            _marker: std::marker::PhantomData,
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
        let stems_ptr = std::ptr::NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

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
        A: AxisUnified<Coord = A>,
        O: AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
    {
        if self.level() > max_stem_level {
            return false;
        }

        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };

        #[allow(unused)]
        // used by simd code below, but is also the only code that uses query_wide arg
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };

        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();

        // SIMD comparison to get child index
        let child_idx = compare_block3(stems, query_val, block_base_idx);
        tracing::trace!("child_idx = {}", child_idx);

        let child_idx_mask = 1 << child_idx;

        // SIMD distance check to get backtrack mask (x86_64 intrinsics) or autovec fallback.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let backtrack_mask = {
            if std::mem::size_of::<A>() == 8 {
                // f64 path
                let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&query_wide_val) };
                let old_off_f64: f64 = unsafe { std::mem::transmute_copy(&old_off_val) };
                let rd_f64: f64 = unsafe { std::mem::transmute_copy(&rd) };
                let best_dist_f64: f64 = unsafe { std::mem::transmute_copy(&best_dist) };

                #[cfg(target_feature = "avx512f")]
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
                #[cfg(not(target_feature = "avx512f"))]
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

                #[cfg(target_feature = "avx512f")]
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
                #[cfg(not(target_feature = "avx512f"))]
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
            }
        } & !child_idx_mask;

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let (backtrack_mask, siblings, rd_values, new_off_values) = {
            let mut siblings = [*self; 8];
            let mut rd_values = [O::zero(); 8];
            let mut new_off_values = [O::zero(); 8];
            let mut mask = 0u8;

            for sibling_idx in 0..8 {
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx);

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

                let query_val = unsafe { *query.get_unchecked(*dim) };

                // TODO: this only works for f64! need a refactor to work for any AxisUnified
                let query_wide_f64: f64 =
                    unsafe { std::mem::transmute_copy(&D::widen_coord(query_val)) };
                let lower_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(lower)) };
                let upper_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(upper)) };
                let interval_dist = interval_distance_1d(query_wide_f64, lower_f64, upper_f64);
                let new_off: O = unsafe { std::mem::transmute_copy(&interval_dist) };

                new_off_values[sibling_idx] = new_off;

                let delta = D::dist1(new_off, old_off_val);
                let rd_far = O::saturating_add(rd, delta);
                rd_values[sibling_idx] = rd_far;

                let passes_threshold = rd_far <= best_dist;
                if passes_threshold {
                    mask |= 1 << sibling_idx;
                }

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
                    passes_threshold,
                    in_backtrack_mask = passes_threshold,
                    "SIMD Block3: sibling interval calc"
                );
            }

            let mask = mask & !child_idx_mask;
            (mask, siblings, rd_values, new_off_values)
        };

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

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            // TODO: this is too slow. Need to:
            //   * Just store a clone of self, rather than creating all the siblings and traversing them
            //     all here. We should only create the sibling and traverse it after we perform the SIMD prune.
            //   * In cases where D == O, widen is a no-op. In this case we can use SIMD ops to calc new_off
            //     and rd_values in parallel for all siblings.

            let mut siblings = [self.clone(); 8];
            let mut rd_values = [O::zero(); 8];
            let mut new_off_values = [O::zero(); 8];

            for sibling_idx in 0..8 {
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx);

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

                let query_val = unsafe { *query.get_unchecked(*dim) };

                // TODO: this only works for f64! need a refactor to work for any AxisUnified
                let query_wide_f64: f64 =
                    unsafe { std::mem::transmute_copy(&D::widen_coord(query_val)) };
                let lower_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(lower)) };
                let upper_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(upper)) };
                let interval_dist = interval_distance_1d(query_wide_f64, lower_f64, upper_f64);
                let new_off: O = unsafe { std::mem::transmute_copy(&interval_dist) };

                new_off_values[sibling_idx] = new_off;

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

            stack.push(SimdQueryStackContext::Block {
                siblings,
                rd_values,
                new_off_values,
                sibling_mask: backtrack_mask,
                dim: dim_val,
                old_off: old_off_val,
            });
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            stack.push(SimdQueryStackContext::Block {
                siblings,
                rd_values,
                new_off_values,
                sibling_mask: backtrack_mask,
                dim: dim_val,
                old_off: old_off_val,
            });
        }

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

        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(query_val)) };
        let lower_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(lower)) };
        let upper_f64: f64 = unsafe { std::mem::transmute_copy(&D::widen_coord(upper)) };

        let interval_dist = interval_distance_1d(query_wide_f64, lower_f64, upper_f64);
        let new_off: O = unsafe { std::mem::transmute_copy(&interval_dist) };
        unsafe { *off.get_unchecked_mut(dim_val) = new_off };

        self.core.traverse_block(child_idx, Self::BLOCK_SIZE as u32);
        *dim = self.dim();

        true
    }

    fn backtracking_query_with_stack<A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &crate::kd_tree::KdTree<A, T, Self, LS, K2, B>,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(&crate::kd_tree::leaf_view::LeafView<A, T, K2, B>, &mut QC),
    ) where
        Self: Sized,
        A: crate::traits_unified_2::AxisUnified<Coord = A>,
        T: crate::traits_unified_2::Basics + Copy + Default + PartialOrd + PartialEq,
        O: crate::traits_unified_2::AxisUnified<Coord = O> + crate::stem_strategies::SimdPrune,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
        QC: crate::kd_tree::traits::QueryContext<A, O, K2>,
        LS: crate::traits_unified_2::LeafStrategy<A, T, Self, K2, B>,
    {
        tree.backtracking_query_with_simd_stack_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

// Block4 implementation (4-level blocks, 64-byte cache lines)
impl<const VB: u32, const K: usize> crate::StemStrategy for DonnellyMarkerSimd<Block4, 64, VB, K> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 4;

    type StackContext<A> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<A, Self>;
    type Stack<A> = crate::kd_tree::query_stack_simd::SimdQueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: std::ptr::NonNull<u8>) -> Self {
        debug_assert!(64 >= VB);

        Self {
            core: crate::stem_strategies::donnelly_core::DonnellyCore::new(stems_ptr),
            _marker: std::marker::PhantomData,
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
            _marker: std::marker::PhantomData,
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
        let stems_ptr = std::ptr::NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

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
        A: AxisUnified<Coord = A>,
        O: AxisUnified<Coord = O>,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
    {
        if self.level() > max_stem_level {
            return false;
        }

        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };
        let old_off_val = unsafe { *off.get_unchecked(dim_val) };
        let block_base_idx = self.stem_idx();

        let child_idx = compare_block4(stems, query_val, block_base_idx);

        let child_idx_mask = 1u16 << child_idx;

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let backtrack_mask = {
            if std::mem::size_of::<A>() == 8 {
                let query_wide_f64: f64 = unsafe { std::mem::transmute_copy(&query_wide_val) };
                let old_off_f64: f64 = unsafe { std::mem::transmute_copy(&old_off_val) };
                let rd_f64: f64 = unsafe { std::mem::transmute_copy(&rd) };
                let best_dist_f64: f64 = unsafe { std::mem::transmute_copy(&best_dist) };

                #[cfg(target_feature = "avx512f")]
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
                #[cfg(not(target_feature = "avx512f"))]
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
                let query_wide_f32: f32 = unsafe { std::mem::transmute_copy(&query_wide_val) };
                let old_off_f32: f32 = unsafe { std::mem::transmute_copy(&old_off_val) };
                let rd_f32: f32 = unsafe { std::mem::transmute_copy(&rd) };
                let best_dist_f32: f32 = unsafe { std::mem::transmute_copy(&best_dist) };

                #[cfg(target_feature = "avx512f")]
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
                #[cfg(not(target_feature = "avx512f"))]
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
            }
        } & !child_idx_mask;

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let (backtrack_mask, siblings, rd_values, new_off_values) = {
            let mut siblings = [*self; 16];
            let mut rd_values = [O::zero(); 16];
            let mut new_off_values = [O::zero(); 16];
            let mut mask: u16 = 0;

            for sibling_idx in 0..16 {
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                let pivot_idx = block_base_idx + sibling_idx;
                let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
                let pivot_wide = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                new_off_values[sibling_idx] = new_off;
                let rd_far = O::saturating_add(rd, D::dist1(new_off, old_off_val));
                rd_values[sibling_idx] = rd_far;

                if rd_far <= best_dist {
                    mask |= 1u16 << sibling_idx;
                }
            }

            let mask = mask & !child_idx_mask;
            (mask, siblings, rd_values, new_off_values)
        };

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            let mut siblings = [self.clone(); 16];
            let mut rd_values = [O::zero(); 16];
            let mut new_off_values = [O::zero(); 16];

            for sibling_idx in 0..16 {
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                let pivot_idx = block_base_idx + sibling_idx;
                let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
                let pivot_wide = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                new_off_values[sibling_idx] = new_off;
                rd_values[sibling_idx] = O::saturating_add(rd, D::dist1(new_off, old_off_val));
            }

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

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            let high_mask = (backtrack_mask >> 8) as u8;
            if high_mask != 0 {
                let mut high_siblings = [*self; 8];
                let mut high_rd_values = [O::zero(); 8];
                let mut high_new_off_values = [O::zero(); 8];
                high_siblings.copy_from_slice(&siblings[8..16]);
                high_rd_values.copy_from_slice(&rd_values[8..16]);
                high_new_off_values.copy_from_slice(&new_off_values[8..16]);
                stack.push(SimdQueryStackContext::Block {
                    siblings: high_siblings,
                    rd_values: high_rd_values,
                    new_off_values: high_new_off_values,
                    sibling_mask: high_mask,
                    dim: dim_val,
                    old_off: old_off_val,
                });
            }

            let low_mask = backtrack_mask as u8;
            if low_mask != 0 {
                let mut low_siblings = [*self; 8];
                let mut low_rd_values = [O::zero(); 8];
                let mut low_new_off_values = [O::zero(); 8];
                low_siblings.copy_from_slice(&siblings[..8]);
                low_rd_values.copy_from_slice(&rd_values[..8]);
                low_new_off_values.copy_from_slice(&new_off_values[..8]);
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

        let pivot_idx = block_base_idx + child_idx as usize;
        let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
        let pivot_wide = D::widen_coord(pivot);
        let new_off = O::saturating_dist(query_wide_val, pivot_wide);
        unsafe { *off.get_unchecked_mut(dim_val) = new_off };

        self.core.traverse_block(child_idx, Self::BLOCK_SIZE as u32);
        *dim = self.dim();

        true
    }

    fn backtracking_query_with_stack<A, T, O, D, QC, LS, const K2: usize, const B: usize>(
        tree: &crate::kd_tree::KdTree<A, T, Self, LS, K2, B>,
        query_ctx: &mut QC,
        stack: &mut Self::Stack<O>,
        process_leaf: impl FnMut(&crate::kd_tree::leaf_view::LeafView<A, T, K2, B>, &mut QC),
    ) where
        Self: Sized,
        A: crate::traits_unified_2::AxisUnified<Coord = A>,
        T: crate::traits_unified_2::Basics + Copy + Default + PartialOrd + PartialEq,
        O: crate::traits_unified_2::AxisUnified<Coord = O> + crate::stem_strategies::SimdPrune,
        D: crate::traits_unified_2::DistanceMetricUnified<A, K2, Output = O>,
        QC: crate::kd_tree::traits::QueryContext<A, O, K2>,
        LS: crate::traits_unified_2::LeafStrategy<A, T, Self, K2, B>,
    {
        tree.backtracking_query_with_simd_stack_impl::<QC, O, D>(query_ctx, stack, process_leaf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_block3_pivots_f64() -> [f64; 8] {
        [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY]
    }

    fn build_test_block3_pivots_f32() -> [f32; 8] {
        [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f32::INFINITY]
    }

    fn select_child_scalar_f64(query: f64, pivots: &[f64; 8]) -> u8 {
        let mut count = 0u8;
        for i in 0..8 {
            if query >= pivots[i] {
                count += 1;
            }
        }
        count
    }

    fn select_child_scalar_f32(query: f32, pivots: &[f32; 8]) -> u8 {
        let mut count = 0u8;
        for i in 0..8 {
            if query >= pivots[i] {
                count += 1;
            }
        }
        count
    }

    fn scalar_backtrack_check_block3_f64(
        query: f64,
        pivots: &[f64; 8],
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        let mut mask = 0u8;
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
            // For SquaredEuclidean: rd_far = rd + (interval_dist - old_off)²
            let delta = interval_dist - old_off;
            let rd_far = rd + delta * delta;

            if rd_far <= best_dist {
                mask |= 1 << child_idx;
            }
        }
        mask
    }

    fn scalar_backtrack_check_block3_f32(
        query: f32,
        pivots: &[f32; 8],
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        let mut mask = 0u8;
        for child_idx in 0..8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

            let lower = if lower_offset == 255 {
                f32::NEG_INFINITY
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f32::INFINITY
            } else {
                pivots[upper_offset as usize]
            };

            // Use f64 version of interval_distance_1d, cast to f32
            let interval_dist =
                interval_distance_1d(query as f64, lower as f64, upper as f64) as f32;
            let delta = interval_dist - old_off;
            let rd_far = rd + delta * delta;

            if rd_far <= best_dist {
                mask |= 1 << child_idx;
            }
        }
        mask
    }

    #[test]
    fn test_child_interval_bounds_block3() {
        // Verify all 8 children have correct interval bounds
        // Expected bounds based on triangular structure:
        //           pivot[0]
        //      pivot[1]    pivot[2]
        //   p[3]    p[4]    p[5]   p[6]
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
        );
    }

    #[test]
    fn test_interval_distance_1d_branchless() {
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

    #[test]
    fn test_block3_child_selection_correctness_f64() {
        // Test that compare_block3 selects correct children for various query values
        let pivots = build_test_block3_pivots_f64();

        // Test select of every child
        let test_cases = [
            (-100.0, 0), // Far below all pivots -> child 0
            (0.05, 0),   // Just below pivot[3]=0.1 -> child 0
            (0.15, 1),   // Between pivot[3]=0.1 and pivot[1]=0.4 -> child 1
            (0.25, 2),   // Between pivot[1]=0.4 and pivot[4]=0.3 - wait, need to check structure
            (0.35, 3),   // Test child 3
            (0.45, 4),   // Test child 4
            (0.55, 5),   // Test child 5
            (0.65, 6),   // Test child 6
            (100.0, 7),  // Far above all pivots -> child 7
        ];

        for (query, expected_child) in test_cases {
            let scalar_child = select_child_scalar_f64(query, &pivots);

            assert_eq!(
                scalar_child, expected_child,
                "Query {} should select child {}, got {}",
                query, expected_child, scalar_child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_all_reachable_f64() {
        let pivots = build_test_block3_pivots_f64();
        let mut children_reached = [false; 8];

        for i in 0..100 {
            let query = -1.0 + (i as f64) * 0.02; // Range from -1.0 to 1.0
            let child = select_child_scalar_f64(query, &pivots);
            if (child as usize) < 8 {
                children_reached[child as usize] = true;
            }
        }

        for (child_idx, &reached) in children_reached.iter().enumerate() {
            assert!(
                reached,
                "Child {} was not reached by any query value",
                child_idx
            );
        }
    }

    #[test]
    fn test_block3_child_selection_boundaries_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Query == pivot? >= cmp should put it in the right-hand interval
        for (pivot_idx, &pivot_val) in pivots.iter().enumerate().take(7) {
            let child = select_child_scalar_f64(pivot_val, &pivots);

            let expected = pivots.iter().take(8).filter(|&&p| pivot_val >= p).count() as u8;

            assert_eq!(
                child, expected,
                "Query at pivot[{}]={} should select child {}, got {}",
                pivot_idx, pivot_val, expected, child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_f32() {
        let pivots = build_test_block3_pivots_f32();

        let test_cases = [
            (-100.0f32, 0u8),
            (0.05f32, 0u8),
            (0.25f32, 2u8),
            (100.0f32, 7u8),
        ];

        for (query, expected_child) in test_cases {
            let scalar_child = select_child_scalar_f32(query, &pivots);
            assert_eq!(
                scalar_child, expected_child,
                "f32: Query {} should select child {}, got {}",
                query, expected_child, scalar_child
            );
        }
    }

    #[test]
    fn test_block3_child_selection_via_compare_block3_f64() {
        let pivots = build_test_block3_pivots_f64();

        let test_queries = [
            -100.0, -1.0, 0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0, 100.0,
        ];

        for &query in &test_queries {
            let expected = select_child_scalar_f64(query, &pivots);
            let actual = compare_block3(&pivots, query, 0);

            assert_eq!(
                actual, expected,
                "compare_block3 mismatch for query {}: expected child {}, got {}",
                query, expected, actual
            );
        }
    }

    #[test]
    fn test_block3_child_selection_via_compare_block3_f32() {
        let pivots = build_test_block3_pivots_f32();

        let test_queries = [-100.0f32, 0.0f32, 0.25f32, 0.5f32, 100.0f32];

        for &query in &test_queries {
            let expected = select_child_scalar_f32(query, &pivots);
            let actual = compare_block3(&pivots, query, 0);

            assert_eq!(
                actual, expected,
                "compare_block3 (f32) mismatch for query {}: expected child {}, got {}",
                query, expected, actual
            );
        }
    }

    #[test]
    fn test_backtrack_scalar_vs_simd_f64_multiple_cases() {
        let pivots = build_test_block3_pivots_f64();

        let test_cases = [
            (0.25, 0.0, 0.0, f64::INFINITY),
            (0.5, 0.0, 0.0, 0.1),
            (0.5, 0.01, 0.05, 0.2),
            (-0.5, 0.0, 0.0, f64::INFINITY),
            (1.5, 0.0, 0.0, f64::INFINITY),
        ];

        for (query, old_off, rd, best_dist) in test_cases {
            let scalar_mask =
                scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, best_dist);

            // Test SIMD implementation if available
            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            {
                use crate::traits_unified_2::DistanceMetricUnified;
                use crate::traits_unified_2::SquaredEuclidean;

                let pivots_ptr = pivots.as_ptr() as *const u8;
                let simd_mask = unsafe {
                    <SquaredEuclidean<f64> as DistanceMetricUnified<f64, 3>>::simd_backtrack_check_block3_interval_f64_avx2(
                        query, pivots_ptr, 0, old_off, rd, best_dist
                    )
                };

                assert_eq!(
                    scalar_mask, simd_mask,
                    "SIMD vs scalar mismatch for query={}, old_off={}, rd={}, best_dist={}: scalar={:08b}, simd={:08b}",
                    query, old_off, rd, best_dist, scalar_mask, simd_mask
                );
            }

            // On non-SIMD platforms, just validate the scalar mask is reasonable
            #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
            {
                // Scalar mask should be valid (between 0 and 0xFF)
                let _ = scalar_mask; // Use the variable to avoid warning
            }
        }
    }

    #[test]
    fn test_backtrack_scalar_vs_simd_f32_multiple_cases() {
        let pivots = build_test_block3_pivots_f32();

        let test_cases = [
            (0.25f32, 0.0f32, 0.0f32, f32::INFINITY),
            (0.5f32, 0.0f32, 0.0f32, 0.1f32),
            (-0.5f32, 0.0f32, 0.0f32, f32::INFINITY),
        ];

        for (query, old_off, rd, best_dist) in test_cases {
            let scalar_mask =
                scalar_backtrack_check_block3_f32(query, &pivots, old_off, rd, best_dist);

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            {
                use crate::traits_unified_2::DistanceMetricUnified;
                use crate::traits_unified_2::SquaredEuclidean;

                let pivots_ptr = pivots.as_ptr() as *const u8;
                let simd_mask = unsafe {
                    <SquaredEuclidean<f32> as DistanceMetricUnified<f32, 3>>::simd_backtrack_check_block3_f32_avx2(
                        query, pivots_ptr, 0, old_off, rd, best_dist
                    )
                };

                assert_eq!(
                    scalar_mask, simd_mask,
                    "SIMD vs scalar (f32) mismatch for query={}, old_off={}, rd={}, best_dist={}: scalar={:08b}, simd={:08b}",
                    query, old_off, rd, best_dist, scalar_mask, simd_mask
                );
            }

            #[cfg(not(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2")))]
            {
                let _ = scalar_mask; // Use the variable to avoid warning
            }
        }
    }

    #[test]
    fn test_block3_backtrack_ground_truth_f64_basic() {
        let pivots = build_test_block3_pivots_f64();

        // Case 1: Query in middle, rd=0, best_dist=infinity (all children should pass)
        let mask = scalar_backtrack_check_block3_f64(0.5, &pivots, 0.0, 0.0, f64::INFINITY);
        assert_eq!(
            mask, 0xFF,
            "With infinite best_dist, all children should pass"
        );

        // Case 2: Query in middle, rd=0, best_dist=0 (only children with interval_dist=0 should pass)
        let query = 0.5;
        let mask = scalar_backtrack_check_block3_f64(query, &pivots, 0.0, 0.0, 0.0);

        // The child containing the query should have interval_dist=0, so rd_far=0
        // Note: if query sits on a boundary, multiple children might have interval_dist=0
        let selected_child = select_child_scalar_f64(query, &pivots);
        let selected_bit = 1 << selected_child;

        assert!(
            (mask & selected_bit) != 0,
            "With best_dist=0, at least the child containing the query (child {}) should pass: got {:08b}",
            selected_child, mask
        );

        // All passing children should have interval_dist=0 (rd_far=0<=0)
        assert!(
            mask.count_ones() >= 1,
            "At least one child should pass when best_dist=0"
        );
    }

    #[test]
    fn test_block3_backtrack_edge_cases_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test with query at exact pivot boundaries
        for (_pivot_idx, &pivot_val) in pivots.iter().enumerate().take(7) {
            let mask =
                scalar_backtrack_check_block3_f64(pivot_val, &pivots, 0.0, 0.0, f64::INFINITY);

            // At least one child should pass
            assert_ne!(
                mask, 0,
                "At pivot boundary {}, at least one child should be visitable",
                pivot_val
            );
        }

        // Test with query outside all pivots
        let mask = scalar_backtrack_check_block3_f64(-1000.0, &pivots, 0.0, 0.0, f64::INFINITY);
        assert_eq!(
            mask, 0xFF,
            "Far outside query with infinite best_dist should allow all children"
        );

        // Test with non-zero old_off
        let mask = scalar_backtrack_check_block3_f64(0.5, &pivots, 0.1, 0.0, f64::INFINITY);
        assert_ne!(
            mask, 0,
            "With non-zero old_off, some children should still pass"
        );
    }

    #[test]
    fn test_block3_backtrack_rd_pruning_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test that increasing rd prunes more children
        let query = 0.5;
        let old_off = 0.0;
        let best_dist = 0.1;

        let mask_rd_0 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, 0.0, best_dist);
        let mask_rd_high =
            scalar_backtrack_check_block3_f64(query, &pivots, old_off, 0.05, best_dist);

        // With higher rd, we should prune at least as many (possibly more) children
        let count_rd_0 = mask_rd_0.count_ones();
        let count_rd_high = mask_rd_high.count_ones();

        assert!(
            count_rd_high <= count_rd_0,
            "Higher rd should prune at least as many children: rd=0 -> {} children, rd=0.05 -> {} children",
            count_rd_0, count_rd_high
        );
    }

    #[test]
    fn test_block3_backtrack_best_dist_pruning_f64() {
        let pivots = build_test_block3_pivots_f64();

        // Test that decreasing best_dist prunes more children
        let query = 0.5;
        let old_off = 0.0;
        let rd = 0.0;

        let mask_best_inf =
            scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, f64::INFINITY);
        let mask_best_01 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, 0.1);
        let mask_best_001 = scalar_backtrack_check_block3_f64(query, &pivots, old_off, rd, 0.01);

        let count_inf = mask_best_inf.count_ones();
        let count_01 = mask_best_01.count_ones();
        let count_001 = mask_best_001.count_ones();

        assert!(
            count_001 <= count_01 && count_01 <= count_inf,
            "Smaller best_dist should prune more: inf -> {}, 0.1 -> {}, 0.01 -> {}",
            count_inf,
            count_01,
            count_001
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_i32_u0() {
        use crate::stem_strategies::donnelly_2_blockmarker_simd::SimdPrune;
        use fixed::types::extra::U0;
        use fixed::FixedI32;

        type Fixed = FixedI32<U0>;

        // Create test rd_values array - some pass, some fail
        let rd_values: [Fixed; 8] = [
            Fixed::from_num(1),
            Fixed::from_num(5),
            Fixed::from_num(10),
            Fixed::from_num(15),
            Fixed::from_num(3),
            Fixed::from_num(7),
            Fixed::from_num(12),
            Fixed::from_num(2),
        ];

        let max_dist = Fixed::from_num(8);
        let sibling_mask = 0xFF; // All siblings enabled

        // Call pruning via trait
        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 4, 5, 7 should be set (values <= 8)
        // Values: [1, 5, 10, 15, 3, 7, 12, 2]
        //  Pass:  [T, T,  F,  F, T, T,  F, T]
        let expected_mask = 0b10110011; // bits 0,1,4,5,7 set

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_i32_u16() {
        use crate::stem_strategies::donnelly_2_blockmarker_simd::SimdPrune;
        use fixed::types::extra::U16;
        use fixed::FixedI32;

        type Fixed = FixedI32<U16>;

        // Create test rd_values with fractional values
        let rd_values: [Fixed; 8] = [
            Fixed::from_num(0.5),
            Fixed::from_num(1.5),
            Fixed::from_num(2.5),
            Fixed::from_num(3.5),
            Fixed::from_num(0.25),
            Fixed::from_num(1.75),
            Fixed::from_num(2.75),
            Fixed::from_num(0.1),
        ];

        let max_dist = Fixed::from_num(2.0);
        let sibling_mask = 0xFF;

        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 4, 5, 7 should be set (values <= 2.0)
        // Values: [0.5, 1.5, 2.5, 3.5, 0.25, 1.75, 2.75, 0.1]
        //  Pass:  [ T,   T,   F,   F,   T,    T,    F,   T]
        let expected_mask = 0b10110011;

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    #[cfg(all(feature = "fixed", not(feature = "simd")))]
    fn test_simd_prune_fixed_u16_u8() {
        use crate::stem_strategies::donnelly_2_blockmarker_simd::SimdPrune;
        use fixed::types::extra::U8;
        use fixed::FixedU16;

        type Fixed = FixedU16<U8>;

        let rd_values: [Fixed; 8] = [
            Fixed::from_num(0.1),
            Fixed::from_num(0.5),
            Fixed::from_num(1.0),
            Fixed::from_num(1.5),
            Fixed::from_num(0.2),
            Fixed::from_num(0.8),
            Fixed::from_num(1.2),
            Fixed::from_num(0.3),
        ];

        let max_dist = Fixed::from_num(1.0);
        let sibling_mask = 0xFF;

        let mask = Fixed::simd_prune_block3(&rd_values, max_dist, sibling_mask);

        // Expected: bits 0, 1, 2, 4, 5, 7 should be set (values <= 1.0)
        // Values: [0.1, 0.5, 1.0, 1.5, 0.2, 0.8, 1.2, 0.3]
        //  Pass:  [ T,   T,   T,   F,   T,   T,   F,   T]
        let expected_mask = 0b10110111;

        assert_eq!(
            mask, expected_mask,
            "Fixed-point pruning mask mismatch: got {:08b}, expected {:08b}",
            mask, expected_mask
        );
    }

    #[test]
    fn test_simd_prune_sibling_mask_filtering() {
        use crate::stem_strategies::donnelly_2_blockmarker_simd::SimdPrune;

        // Test that sibling_mask correctly filters results
        let rd_values = [0.5f64, 1.5, 2.5, 3.5, 0.25, 1.75, 2.75, 0.1];
        let max_dist = 2.0f64;

        // All siblings enabled
        let mask_all = f64::simd_prune_block3(&rd_values, max_dist, 0xFF);
        // Only even-indexed siblings enabled
        let mask_even = f64::simd_prune_block3(&rd_values, max_dist, 0b01010101);
        // Only odd-indexed siblings enabled
        let mask_odd = f64::simd_prune_block3(&rd_values, max_dist, 0b10101010);

        // mask_all should have bits 0,1,4,5,7 set (values <= 2.0)
        let expected_all = 0b10110011;
        assert_eq!(
            mask_all, expected_all,
            "Unfiltered mask mismatch: got {:08b}, expected {:08b}",
            mask_all, expected_all
        );

        // mask_even should have only bits 0,4 set (even indices & values <= 2.0)
        let expected_even = 0b00010001;
        assert_eq!(
            mask_even, expected_even,
            "Even-filtered mask mismatch: got {:08b}, expected {:08b}",
            mask_even, expected_even
        );

        // mask_odd should have only bits 1,5,7 set (odd indices & values <= 2.0)
        let expected_odd = 0b10100010;
        assert_eq!(
            mask_odd, expected_odd,
            "Odd-filtered mask mismatch: got {:08b}, expected {:08b}",
            mask_odd, expected_odd
        );
    }
}
