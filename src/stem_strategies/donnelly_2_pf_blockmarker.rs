//! V2 Donnelly Stem Strategy with Prefetch, refactored from
//! const generic block size to marker trait

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::traits_unified_2::AxisUnified;
use crate::StemStrategy;

use crate::stem_strategies::donnelly_core::DonnellyCore;
use crate::stem_strategies::{Block3, Block4, Block5, Block6, Block7, BlockSizeMarker};

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

macro_rules! impl_donnelly_stem_strategy {
    ($marker:ty, $size:tt) => {
        impl<const CL: u32, const VB: u32, const K: usize> StemStrategy
            for DonnellyMarkerPf<$marker, CL, VB, K>
        {
            const ROOT_IDX: usize = 0;

            type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self>;
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

            fn get_leaf_idx<A: AxisUnified, const K2: usize>(
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
