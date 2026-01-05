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
        // Check if traversing this block would exceed max_stem_level
        // For block strategies, we traverse multiple levels at once, so we need to check
        // if the level AFTER traversing would be beyond the limit
        if self.level() + Self::BLOCK_SIZE as i32 > max_stem_level {
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
                D::simd_backtrack_check_block3_f64_avx2(
                    unsafe { std::mem::transmute_copy(&query_wide_f64) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f64) },
                    unsafe { std::mem::transmute_copy(&rd_f64) },
                    unsafe { std::mem::transmute_copy(&best_dist_f64) },
                )
            }
        } else if std::mem::size_of::<A>() == 4 {
            // f32 path
            let query_wide_f32: f32 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f32: f32 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f32: f32 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f32: f32 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                D::simd_backtrack_check_block3_f32_avx512(
                    unsafe { std::mem::transmute_copy(&query_wide_f32) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f32) },
                    unsafe { std::mem::transmute_copy(&rd_f32) },
                    unsafe { std::mem::transmute_copy(&best_dist_f32) },
                )
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                D::simd_backtrack_check_block3_f32_avx2(
                    unsafe { std::mem::transmute_copy(&query_wide_f32) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f32) },
                    unsafe { std::mem::transmute_copy(&rd_f32) },
                    unsafe { std::mem::transmute_copy(&best_dist_f32) },
                )
            }
        } else {
            panic!("Unsupported axis type size");
        };

        // Remove the path we're taking from backtrack candidates
        let backtrack_mask = backtrack_mask & !(1 << child_idx);

        // If any siblings need exploration, push them as a Block entry
        if backtrack_mask != 0 {
            use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

            // Calculate all sibling contexts and their rd_far values
            let mut siblings = [self.clone(); 8];
            let mut rd_values = [O::zero(); 8];

            for sibling_idx in 0..8 {
                // Create context for this sibling
                siblings[sibling_idx]
                    .core
                    .traverse_block(sibling_idx as u8, Self::BLOCK_SIZE as u32);

                // Calculate rd_far for this sibling
                let pivot_idx = block_base_idx + sibling_idx;
                let pivot = unsafe { *stems.get_unchecked(pivot_idx) };
                let pivot_wide = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                rd_values[sibling_idx] = O::saturating_add(rd, D::dist1(new_off, old_off_val));
            }

            // Push single Block entry containing all siblings
            stack.push(SimdQueryStackContext::Block {
                siblings,
                rd_values,
                sibling_mask: backtrack_mask,
                dim: dim_val,
                old_off: old_off_val,
            });
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
        // Check if traversing this block would exceed max_stem_level
        // For block strategies, we traverse multiple levels at once, so we need to check
        // if the level AFTER traversing would be beyond the limit
        if self.level() + Self::BLOCK_SIZE as i32 > max_stem_level {
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
                D::simd_backtrack_check_block4_f64_avx512(
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
                D::simd_backtrack_check_block4_f64_avx2(
                    unsafe { std::mem::transmute_copy(&query_wide_f64) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f64) },
                    unsafe { std::mem::transmute_copy(&rd_f64) },
                    unsafe { std::mem::transmute_copy(&best_dist_f64) },
                )
            }
        } else if std::mem::size_of::<A>() == 4 {
            // f32 path
            let query_wide_f32: f32 = unsafe { std::mem::transmute_copy(&query_wide_val) };
            let old_off_f32: f32 = unsafe { std::mem::transmute_copy(&old_off_val) };
            let rd_f32: f32 = unsafe { std::mem::transmute_copy(&rd) };
            let best_dist_f32: f32 = unsafe { std::mem::transmute_copy(&best_dist) };

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                D::simd_backtrack_check_block4_f32_avx512(
                    unsafe { std::mem::transmute_copy(&query_wide_f32) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f32) },
                    unsafe { std::mem::transmute_copy(&rd_f32) },
                    unsafe { std::mem::transmute_copy(&best_dist_f32) },
                )
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
            {
                D::simd_backtrack_check_block4_f32_avx2(
                    unsafe { std::mem::transmute_copy(&query_wide_f32) },
                    stems_ptr.as_ptr(),
                    block_base_idx,
                    unsafe { std::mem::transmute_copy(&old_off_f32) },
                    unsafe { std::mem::transmute_copy(&rd_f32) },
                    unsafe { std::mem::transmute_copy(&best_dist_f32) },
                )
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
                rd_values[sibling_idx] = O::saturating_add(rd, D::dist1(new_off, old_off_val));
            }

            // Push high block (siblings 8-15) first, since stack is LIFO
            let high_mask = (backtrack_mask >> 8) as u8;
            if high_mask != 0 {
                let mut high_siblings = [self.clone(); 8];
                let mut high_rd_values = [O::zero(); 8];
                for i in 0..8 {
                    high_siblings[i] = siblings[i + 8].clone();
                    high_rd_values[i] = rd_values[i + 8];
                }
                stack.push(SimdQueryStackContext::Block {
                    siblings: high_siblings,
                    rd_values: high_rd_values,
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
                for i in 0..8 {
                    low_siblings[i] = siblings[i].clone();
                    low_rd_values[i] = rd_values[i];
                }
                stack.push(SimdQueryStackContext::Block {
                    siblings: low_siblings,
                    rd_values: low_rd_values,
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
