use crate::kd_tree::leaf_view::LeafView;
use crate::kd_tree::query_stack::{QueryStack, QueryStackContext, StackTrait};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{
    AxisUnified, Basics, DistanceMetricUnified, LeafStrategy, Mutability,
};
use crate::StemStrategy;
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    pub(crate) fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        LS::Mutability::get_leaf_idx(self, query)
    }

    /// Non-backtracking query
    #[inline]
    pub(crate) fn straight_query<QC, O>(
        &self,
        query_ctx: QC,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>),
    ) where
        QC: QueryContext<A, O, K>,
    {
        let leaf_idx = LS::Mutability::get_leaf_idx(self, query_ctx.query());

        tracing::trace!(%leaf_idx, "processing leaf");
        let leaf_view = self.leaves.leaf_view(leaf_idx);
        process_leaf(&leaf_view);
    }

    /// Get the leaf index for a query (unmapped leaves)
    #[inline]
    pub(crate) fn get_leaf_idx_unmapped(&self, query: &[A; K]) -> usize {
        SS::get_leaf_idx(&self.stems, query, self.max_stem_level)
    }

    /// Get the leaf index for a query (mapped leaves)
    #[inline(always)]
    pub(crate) fn get_leaf_idx_mapped(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level {
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return leaf_idx;
            }

            let pivot = unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool = *unsafe { query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        match &self.stem_leaf_resolution {
            crate::kd_tree::StemLeafResolution::Mapped { leaf_idx_map, .. } => {
                leaf_idx_map[stem_strat.stem_idx()].unwrap().get()
            }
            _ => unreachable!(),
        }
    }

    /// Check if a stem points directly to a leaf
    #[inline(always)]
    pub(crate) fn resolve_terminal_stem(&self, stem_idx: usize) -> Option<usize> {
        match &self.stem_leaf_resolution {
            crate::kd_tree::StemLeafResolution::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => {
                if stem_idx >= *min_stem_leaf_idx {
                    let map_idx = stem_idx - *min_stem_leaf_idx;
                    leaf_idx_map
                        .get(map_idx)
                        .and_then(|opt| opt.map(|n| n.get()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Backtracking query
    #[inline(always)]
    pub(crate) fn backtracking_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
        SS::Stack<O>: StackTrait<O, SS>,
    {
        let mut stack = SS::Stack::<O>::default();
        self.backtracking_query_with_stack::<QC, O, D>(query_ctx, &mut stack, process_leaf);
    }

    /// Backtracking query with explicit stack
    #[inline(always)]
    pub(crate) fn backtracking_query_with_stack<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
        SS::Stack<O>: StackTrait<O, SS>,
    {
        SS::backtracking_query_with_stack::<A, T, O, D, QC, LS, K, B>(
            self,
            query_ctx,
            stack,
            process_leaf,
        );
    }

    /// Implementation of backtracking query with scalar stack.
    /// Called by default StemStrategy::backtracking_query_with_stack implementation.
    /// SIMD strategies override the trait method with custom implementations.
    #[inline(always)]
    pub(crate) fn backtracking_query_with_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut QueryStack<O, SS>,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(QueryStackContext::new(stem_strat));

        while let Some(stack_ctx) = stack.pop() {
            let (mut stem_strat, old_off, rd) = stack_ctx.into_parts();
            let mut dim = stem_strat.dim();
            tracing::trace!(%dim, %old_off, %rd, ?off, "Popped stack context");

            let max_dist = query_ctx.max_dist();
            if O::cmp(rd, max_dist) == std::cmp::Ordering::Greater {
                tracing::trace!(%rd, %max_dist, "SCALAR Prune check: PRUNE");
                continue;
            }
            tracing::trace!(%rd, %max_dist, "SCALAR Prune check: VISIT");

            tracing::trace!("Restoring off[{}]. was {}, now {}", dim, off[dim], old_off);
            off[dim] = old_off;

            let best_dist = query_ctx.max_dist();
            let leaf_idx = self.traverse_to_leaf::<O, D>(
                &query,
                &query_wide,
                &mut stem_strat,
                &mut off,
                &mut dim,
                rd,
                best_dist,
                stack,
            );

            tracing::trace!(%leaf_idx, "processing leaf");
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            process_leaf(&leaf_view, query_ctx);
        }
    }

    /// traverse to leaf
    #[inline(always)]
    fn traverse_to_leaf<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut QueryStack<O, SS>,
    ) -> usize
    where
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        loop {
            // Check if current stem points directly to a leaf
            // For Immutable trees, this should optimise away since resolve_terminal_stem_idx
            // will always return None
            if let Some(leaf_idx) =
                LS::Mutability::resolve_terminal_stem_idx(self, stem_strat.stem_idx())
            {
                return leaf_idx;
            }

            // Delegate to stem strategy for traversal step
            // Default impl does level-by-level, block-based strategies do block-at-once
            // SAFETY: Cast concrete QueryStack to associated type SS::Stack for trait call
            let stack_ref = unsafe { &mut *(stack as *mut QueryStack<O, SS> as *mut SS::Stack<O>) };
            let should_continue = stem_strat.backtracking_traverse_step::<A, O, D, K>(
                &self.stems,
                query,
                query_wide,
                off,
                dim,
                rd,
                self.max_stem_level,
                best_dist,
                stack_ref,
            );

            if !should_continue {
                break;
            }
        }

        stem_strat.leaf_idx()
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellyMarkerSimd's backtracking_query_with_stack override.
    #[cfg(feature = "simd")]
    #[inline(always)]
    pub(crate) fn backtracking_query_with_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(SimdQueryStackContext::new_single(stem_strat));

        // Backtracking loop
        while let Some(ctx) = stack.pop() {
            match ctx {
                SimdQueryStackContext::Single {
                    stem_strat: mut ss,
                    old_off,
                    rd,
                } => {
                    // Single entry - standard scalar processing
                    let mut dim = ss.dim();
                    tracing::trace!(%dim, %old_off, %rd, ?off, "Popped single context");

                    let max_dist = query_ctx.max_dist();
                    if O::cmp(rd, max_dist) == std::cmp::Ordering::Greater {
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }
                    tracing::trace!(%rd, %max_dist, "SIMD Prune check: VISIT");

                    tracing::trace!("Restoring off[{}]. was {}, now {}", dim, off[dim], old_off);
                    off[dim] = old_off;

                    let best_dist = query_ctx.max_dist();
                    let leaf_idx = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    );

                    tracing::trace!(%leaf_idx, "processing leaf");
                    let leaf_view = self.leaves.leaf_view(leaf_idx);
                    process_leaf(&leaf_view, query_ctx);
                }
                SimdQueryStackContext::Block {
                    siblings,
                    rd_values,
                    new_off_values,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                } => {
                    tracing::trace!(%dim_val, %old_off, ?rd_values, %sibling_mask, "Popped block context");

                    // SIMD pruning: check which siblings pass the backtrack test
                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        Self::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        tracing::trace!("All siblings pruned");
                        continue;
                    }

                    tracing::trace!(
                        surviving_mask = format!("{:08b}", surviving_mask),
                        "Some siblings survive"
                    );

                    // Save the current off state before processing siblings
                    let saved_off = off;

                    // Process each surviving sibling
                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = siblings[sibling_idx].clone();
                            let rd = rd_values[sibling_idx];
                            let new_off = new_off_values[sibling_idx];
                            let mut dim = ss.dim();

                            // Restore off array to saved state, then update the split dimension
                            // Use the per-sibling new_off value (e.g., interval distance)
                            off = saved_off;
                            tracing::trace!(
                                "Restoring off[{}]. was {}, now {} (interval dist for sibling {}). Parent dim was {}, sibling dim is {}",
                                dim,
                                off[dim],
                                new_off,
                                sibling_idx,
                                dim_val,
                                dim
                            );
                            off[dim] = new_off;

                            let best_dist = query_ctx.max_dist();
                            let leaf_idx = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            );

                            tracing::trace!(%leaf_idx, "processing leaf");
                            let leaf_view = self.leaves.leaf_view(leaf_idx);
                            process_leaf(&leaf_view, query_ctx);
                        }
                    }
                }
            }
        }
    }

    /// traverse to leaf with SIMD stack
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn traverse_to_leaf_simd<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>,
    ) -> usize
    where
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        loop {
            // Check if current stem points directly to a leaf
            if let Some(leaf_idx) =
                LS::Mutability::resolve_terminal_stem_idx(self, stem_strat.stem_idx())
            {
                return leaf_idx;
            }

            // Delegate to stem strategy for traversal step
            // SAFETY: Cast concrete SimdQueryStack to associated type SS::Stack for trait call
            let stack_ref = unsafe {
                &mut *(stack as *mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>
                    as *mut SS::Stack<O>)
            };
            let should_continue = stem_strat.backtracking_traverse_step::<A, O, D, K>(
                &self.stems,
                query,
                query_wide,
                off,
                dim,
                rd,
                self.max_stem_level,
                best_dist,
                stack_ref,
            );

            tracing::trace!(
                stem_idx = %stem_strat.stem_idx(),
                level = %stem_strat.level(),
                dim = %stem_strat.dim(),
                "Descended one block"
            );

            if !should_continue {
                break;
            }
        }

        stem_strat.leaf_idx()
    }

    /// SIMD prune block helper
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn simd_prune_block<O>(rd_values: &[O; 8], max_dist: O, sibling_mask: u8) -> u8
    where
        O: AxisUnified<Coord = O>,
    {
        if std::mem::size_of::<O>() == 8 {
            // f64 path
            let max_dist_f64: f64 = unsafe { std::mem::transmute_copy(&max_dist) };
            let rd_f64: [f64; 8] = unsafe { std::mem::transmute_copy(rd_values) };
            Self::simd_prune_block_f64(&rd_f64, max_dist_f64, sibling_mask)
        } else if std::mem::size_of::<O>() == 4 {
            // f32 path
            let max_dist_f32: f32 = unsafe { std::mem::transmute_copy(&max_dist) };
            let rd_f32: [f32; 8] = unsafe { std::mem::transmute_copy(rd_values) };
            Self::simd_prune_block_f32(&rd_f32, max_dist_f32, sibling_mask)
        } else {
            panic!("Unsupported output type size");
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn simd_prune_block_f64(rd_values: &[f64; 8], max_dist: f64, sibling_mask: u8) -> u8 {
        unsafe {
            use std::arch::x86_64::*;

            let max_dist_vec = _mm256_set1_pd(max_dist);
            let rd_low = _mm256_loadu_pd(rd_values.as_ptr());
            let rd_high = _mm256_loadu_pd(rd_values.as_ptr().add(4));

            let cmp_low = _mm256_cmp_pd(rd_low, max_dist_vec, _CMP_LE_OQ);
            let cmp_high = _mm256_cmp_pd(rd_high, max_dist_vec, _CMP_LE_OQ);

            let mask_low = _mm256_movemask_pd(cmp_low) as u8;
            let mask_high = _mm256_movemask_pd(cmp_high) as u8;

            let mask = mask_low | (mask_high << 4);

            // We need to account for the fact that the ordering of stem pivots within
            // a 3-block is triangular. e.g.:
            //
            //                               #0 (0.5)
            //            #1 (0.25)                             #2 (0.75)
            // #3 (0.125)          #4 (0.375)        #5 (0.625)          #6 (0.875)

            //  Child Idx |      Val Range      |  Pivot Idx
            //     0      |           x < 0.125 |      3
            //     1      |  0.125 <= x < 0.250 |      1
            //     2      |  0.250 <= x < 0.375 |      4
            //     3      |  0.375 <= x < 0.500 |      0
            //     4      |  0.500 <= x < 0.625 |      5
            //     5      |  0.625 <= x < 0.750 |      2
            //     6      |  0.750 <= x < 0.875 |      6
            //     7      |  0.875 <= x         |

            // Map pivot idx to child idx by permuting the mask

            // Source: https://programming.sirrida.de/calcperm.php
            // Config: LSB First, Origin 0, Base 10, indices refer to source bits
            // Input: "7 3 1 4 0 5 2 6  # bswap"
            // allow all
            // Method used: Bit Group Moving
            // let permuted_mask = (mask & 0x20)
            //     | ((mask & 0x42) << 1)
            //     | ((mask & 0x05) << 4)
            //     | ((mask & 0x80) >> 7)
            //     | ((mask & 0x08) >> 2)
            //     | ((mask & 0x10) >> 1);
            //
            // let masked_permuted_mask = permuted_mask & sibling_mask;
            //
            // masked_permuted_mask

            mask & sibling_mask
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn simd_prune_block_f32(rd_values: &[f32; 8], max_dist: f32, sibling_mask: u8) -> u8 {
        unsafe {
            use std::arch::x86_64::*;

            let max_dist_vec = _mm256_set1_ps(max_dist);
            let rd_vec = _mm256_loadu_ps(rd_values.as_ptr());

            let cmp = _mm256_cmp_ps(rd_vec, max_dist_vec, _CMP_LE_OQ);
            let mask = _mm256_movemask_ps(cmp) as u8;

            mask & sibling_mask
        }
    }
}
