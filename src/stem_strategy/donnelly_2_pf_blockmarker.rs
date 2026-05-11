//! V2 Donnelly Stem Strategy with Prefetch, refactored from
//! const generic block size to marker trait

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::stem_strategy::donnelly_core::DonnellyCore;
use crate::stem_strategy::{Block3, Block4, Block5, Block6, Block7, BlockSizeMarker};
use crate::{Axis, StemStrategy};

/// Donnelly Strategy
///
/// A block-based stem strategy with prefetching, optimized for cache sympathy.
/// Like a non-recursive Van Emde Boas layout.
///
/// - BS: Block size, i.e. minor tri height.
/// - CL: Cache line width in bytes (Most of the time, 64. Can be 128 for Apple M2+)
/// - VB: Value width in bytes (e.g. 4 for f32, 8 for f64)
#[derive(Copy, Clone, Debug)]
pub struct DonnellyMarkerPf<BS: BlockSizeMarker, const CL: u32, const VB: u32, const K: usize> {
    core: DonnellyCore<CL, VB, K>,
    _marker: PhantomData<BS>,
}

/// Scalar block-marker Donnelly strategy using block-sized dimension scheduling.
#[derive(Copy, Clone, Debug)]
pub struct DonnellyMarkerScalar<BS: BlockSizeMarker, const CL: u32, const VB: u32, const K: usize> {
    core: DonnellyCore<CL, VB, K>,
    _marker: PhantomData<BS>,
}

macro_rules! impl_donnelly_stem_strategy {
    ($marker:ty, $size:tt) => {
        impl<const CL: u32, const VB: u32, const K: usize> StemStrategy
            for DonnellyMarkerPf<$marker, CL, VB, K>
        {
            const ROOT_IDX: usize = 0;

            type DeferredState = Self;
            type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
            type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

            #[inline(always)]
            fn new(stems_ptr: NonNull<u8>) -> Self {
                Self {
                    core: DonnellyCore::new(stems_ptr),
                    _marker: PhantomData,
                }
            }

            #[inline(always)]
            fn stem_idx(&self) -> usize { self.core.stem_idx() }

            #[inline(always)]
            fn deferred_state(&self) -> Self::DeferredState { *self }

            #[inline(always)]
            fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) { *self = state; }

            #[inline(always)]
            fn leaf_idx(&self) -> usize { self.core.leaf_idx() }

            // TODO: I temporarily changed this strat so that
            //  it only changes dim once every 3 levels. This
            //  meant I could compare how it processed and backtracked
            //  to the newer SIMD variant so I could see where the
            //  SIMD one was going wrong.
            //  Revert these once happy with SIMD behaviour
            #[inline(always)]
            fn dim(&self) -> usize { self.core.dim() }
            // #[inline(always)]
            // fn dim(&self) -> usize {
            //     self.core.level() as usize / 3 % K
            // }
            // #[inline(always)]
            // fn construction_dim(&self) -> usize {
            //     self.core.level() as usize / 3 % K
            // }

            #[inline(always)]
            fn level(&self) -> i32 { self.core.level() }

            #[inline(always)]
            fn traverse(&mut self, is_right: bool) {
                self.core.traverse(is_right)
            }

            #[inline(always)]
            fn traverse_head(&mut self, is_right: bool) {
                self.core.traverse_head(is_right)
            }

            #[inline(always)]
            fn traverse_tail(&mut self, is_right: bool) {
                self.core.traverse_tail_with_block_size(is_right, $size)
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

            fn get_stem_node_count_from_leaf_node_count(_leaf_node_count: usize) -> usize {
                unimplemented!()
            }
            fn stem_node_padding_factor() -> usize {
                unimplemented!()
            }

            fn block_size() -> usize { $size }

            fn get_leaf_idx<A: Axis, const K2: usize>(
                stems: &[A],
                query: &[A; K2],
                max_stem_level: i32,
            ) -> usize {
                let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut strat = Self::new(stems_ptr);

                while strat.level() + ($size as i32) <= max_stem_level + 1 {
                    impl_donnelly_stem_strategy!(@unroll $size, stems, query, strat);
                }

                while strat.level() <= max_stem_level {
                    let pivot = unsafe { stems.get_unchecked(strat.stem_idx()) };
                    let is_right = unsafe { *query.get_unchecked(strat.dim()) } >= *pivot;
                    strat.traverse(is_right);
                }

                strat.leaf_idx()
            }
        }

        impl<const CL: u32, const VB: u32, const K: usize> StemStrategy
            for DonnellyMarkerScalar<$marker, CL, VB, K>
        {
            const ROOT_IDX: usize = 0;

            type DeferredState = Self;
            type StackContext<A> =
                crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
            type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

            #[inline(always)]
            fn new(stems_ptr: NonNull<u8>) -> Self {
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
            fn deferred_state(&self) -> Self::DeferredState {
                *self
            }

            #[inline(always)]
            fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
                *self = state;
            }

            #[inline(always)]
            fn leaf_idx(&self) -> usize {
                self.core.leaf_idx()
            }

            #[inline(always)]
            fn dim(&self) -> usize {
                self.core.level() as usize / $size % K
            }

            #[inline(always)]
            fn construction_dim(&self) -> usize {
                self.core.level() as usize / $size % K
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
            fn traverse_head(&mut self, is_right: bool) {
                self.core.traverse_head(is_right)
            }

            #[inline(always)]
            fn traverse_tail(&mut self, is_right: bool) {
                self.core.traverse_tail_with_block_size(is_right, $size)
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

            fn get_stem_node_count_from_leaf_node_count(_leaf_node_count: usize) -> usize {
                unimplemented!()
            }

            fn stem_node_padding_factor() -> usize {
                unimplemented!()
            }

            fn block_size() -> usize {
                $size
            }

            fn get_leaf_idx<A: Axis, const K2: usize>(
                stems: &[A],
                query: &[A; K2],
                max_stem_level: i32,
            ) -> usize {
                let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
                let mut strat = Self::new(stems_ptr);

                while strat.level() + ($size as i32) <= max_stem_level + 1 {
                    impl_donnelly_stem_strategy!(@unroll $size, stems, query, strat);
                }

                while strat.level() <= max_stem_level {
                    let pivot = unsafe { stems.get_unchecked(strat.stem_idx()) };
                    let is_right = unsafe { *query.get_unchecked(strat.dim()) } >= *pivot;
                    strat.traverse(is_right);
                }

                strat.leaf_idx()
            }
        }
    };

    (@unroll 3, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 4, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 5, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 6, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@tail $stems, $query, $strat);
    };
    (@unroll 7, $stems:expr, $query:expr, $strat:expr) => {
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@head $stems, $query, $strat);
        impl_donnelly_stem_strategy!(@tail $stems, $query, $strat);
    };

    (@head $stems:expr, $query:expr, $strat:expr) => {{
        let pivot = unsafe { $stems.get_unchecked($strat.stem_idx()) };
        let is_right = unsafe { *$query.get_unchecked($strat.dim()) } >= *pivot;
        $strat.traverse_head(is_right);
    }};
    (@tail $stems:expr, $query:expr, $strat:expr) => {{
        let pivot = unsafe { $stems.get_unchecked($strat.stem_idx()) };
        let is_right = unsafe { *$query.get_unchecked($strat.dim()) } >= *pivot;
        $strat.traverse_tail(is_right);
    }};
}

impl_donnelly_stem_strategy!(Block3, 3);
impl_donnelly_stem_strategy!(Block4, 4);
impl_donnelly_stem_strategy!(Block5, 5);
impl_donnelly_stem_strategy!(Block6, 6);
impl_donnelly_stem_strategy!(Block7, 7);

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    type Block3Scalar = DonnellyMarkerScalar<Block3, 64, 8, 3>;

    #[test]
    fn donnelly_marker_scalar_block3_basics_and_state_round_trip() {
        let mut stems = [f64::INFINITY; 256];
        let stems_ptr = NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap();

        let mut strat = Block3Scalar::new(stems_ptr);
        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.leaf_idx(), 0);
        assert_eq!(strat.dim(), 0);
        assert_eq!(strat.construction_dim(), 0);
        assert_eq!(strat.level(), 0);
        assert_eq!(Block3Scalar::block_size(), 3);

        let saved = strat.deferred_state();
        strat.traverse(true);
        strat.traverse(false);
        assert_ne!(strat.stem_idx(), saved.stem_idx());
        assert_ne!(strat.leaf_idx(), saved.leaf_idx());

        strat.rehydrate_deferred_state(saved);
        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.leaf_idx(), 0);
        assert_eq!(strat.dim(), 0);
        assert_eq!(strat.construction_dim(), 0);
        assert_eq!(strat.level(), 0);
    }

    #[test]
    fn donnelly_marker_scalar_block3_traversal_branch_and_children_match_core() {
        let mut stems = [0.0f64; 256];
        let stems_ptr = NonNull::new(stems.as_mut_ptr() as *mut u8).unwrap();

        let mut wrapper = Block3Scalar::new(stems_ptr);
        let mut core = DonnellyCore::<64, 8, 3>::new(stems_ptr);

        wrapper.traverse_head(true);
        core.traverse_head(true);
        assert_eq!(wrapper.stem_idx(), core.stem_idx());
        assert_eq!(wrapper.leaf_idx(), core.leaf_idx());
        assert_eq!(wrapper.level(), core.level());
        assert_eq!(wrapper.dim(), core.level() as usize / 3 % 3);
        assert_eq!(wrapper.construction_dim(), core.level() as usize / 3 % 3);

        wrapper.traverse_tail(false);
        core.traverse_tail_with_block_size(false, 3);
        assert_eq!(wrapper.stem_idx(), core.stem_idx());
        assert_eq!(wrapper.leaf_idx(), core.leaf_idx());
        assert_eq!(wrapper.level(), core.level());
        assert_eq!(wrapper.child_indices(), core.child_indices());

        let mut branched_wrapper = Block3Scalar::new(stems_ptr);
        let mut branched_core = DonnellyCore::<64, 8, 3>::new(stems_ptr);
        let right_wrapper = branched_wrapper.branch();
        let right_core = branched_core.branch();

        assert_eq!(branched_wrapper.stem_idx(), branched_core.stem_idx());
        assert_eq!(branched_wrapper.leaf_idx(), branched_core.leaf_idx());
        assert_eq!(right_wrapper.stem_idx(), right_core.stem_idx());
        assert_eq!(right_wrapper.leaf_idx(), right_core.leaf_idx());
        assert_eq!(right_wrapper.level(), right_core.level());
        assert_eq!(right_wrapper.dim(), right_core.level() as usize / 3 % 3);
    }

    #[test]
    fn donnelly_marker_scalar_block3_get_leaf_idx_uses_unrolled_and_scalar_paths() {
        let stems = [0.0f64; 256];
        let query = [1.0f64, 1.0, 1.0];

        let leaf_idx = Block3Scalar::get_leaf_idx(&stems, &query, 4);

        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut manual = Block3Scalar::new(stems_ptr);
        while manual.level() <= 4 {
            manual.traverse(true);
        }

        assert_eq!(leaf_idx, manual.leaf_idx());
        assert_eq!(leaf_idx, 31);
    }

    #[test]
    fn donnelly_marker_scalar_block3_unimplemented_metadata_helpers_panic() {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = Block3Scalar::get_stem_node_count_from_leaf_node_count(8);
        }))
        .is_err());

        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = Block3Scalar::stem_node_padding_factor();
        }))
        .is_err());
    }
}
