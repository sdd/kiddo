use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTreeQueryOps;
use crate::kd_tree::{ArchivedKdTree, KdTreeAccessor};
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::best_n_within::best_n_within_with_query_wide_arena;
use crate::leaf_view_chunked::nearest_n_within::nearest_n_within_with_query_wide_arena;
use crate::leaf_view_chunked::nearest_one::{
    nearest_one_with_query_wide, nearest_one_with_query_wide_arena,
};
use crate::results::result_collection::{
    BestNeighbourResultCollection, BinaryHeapResultCollection, ResultCollection,
    VisitorResultCollection,
};
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
// use crate::traits::Content;
use crate::traits_unified_2::{AxisUnified, Basics, LeafProjection, LeafStrategy};
use crate::{BestNeighbour, NearestNeighbour, StemStrategy};
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + AxisUnified<Coord = A> + 'static,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn process_leaf_nearest_one<D>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + 'static,
    {
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                nearest_one_with_query_wide_arena::<A, T, D, K>(
                    &arena, query_wide, best_dist, best_item,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                nearest_one_with_query_wide::<A, T, D, K, B>(
                    &leaf, query_wide, best_dist, best_item,
                );
            }
        }
    }

    /// Finds an approximate nearest point to the query point.
    #[inline(always)]
    pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K, Output = A>,
    {
        let req_ctx = ArchivedApproxNearestOneReqCtx::<A, D::Output, K> {
            query,
            _phantom: std::marker::PhantomData,
        };

        let mut best_dist = A::max_value();
        let mut best_item = T::default();

        self.straight_query(req_ctx, |leaf_idx| {
            self.process_leaf_nearest_one::<D>(leaf_idx, query, &mut best_dist, &mut best_item);
        });

        (best_dist, best_item)
    }

    /// Finds the nearest point to the query point.
    #[inline(always)]
    pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static,
    {
        let mut req_ctx = ArchivedNearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, query_ctx| {
            self.process_leaf_nearest_one::<D>(
                leaf_idx,
                query_wide,
                &mut query_ctx.best_dist,
                &mut query_ctx.best_item,
            );
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + AxisUnified<Coord = A> + 'static,
    T: Basics + PartialOrd,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn process_leaf_nearest_n_within<D, R>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
        R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
    {
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, R, K>(
                    &arena, query_wide, max_dist, results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                crate::leaf_view_chunked::nearest_n_within::nearest_n_within_with_query_wide::<
                    A,
                    T,
                    D,
                    R,
                    K,
                    B,
                >(&leaf, query_wide, max_dist, results);
            }
        }
    }

    /// Finds up to N nearest points within a given distance.
    pub fn nearest_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let max_qty = max_qty.get();
        if max_qty == usize::MAX {
            self.nearest_n_within_inner::<D, Vec<NearestNeighbour<D::Output, T>>>(
                query, max_dist, max_qty, sorted,
            )
        } else {
            self.nearest_n_within_inner::<
                D,
                BinaryHeapResultCollection<NearestNeighbour<D::Output, T>>,
            >(query, max_dist, max_qty, sorted)
        }
    }

    fn nearest_n_within_inner<D, R>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut req_ctx = ArchivedNearestNWithinReqCtx {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let leaf_max_dist = req_ctx.max_dist();
            self.process_leaf_nearest_n_within::<D, R>(
                leaf_idx,
                query_wide,
                leaf_max_dist,
                &mut req_ctx.results,
            );
        });

        if sorted {
            req_ctx.results.into_sorted_vec()
        } else {
            req_ctx.results.into_vec()
        }
    }

    /// Finds the N nearest points to the query point.
    pub fn nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within::<D>(query, D::Output::max_value(), max_qty, sorted)
    }

    /// Finds all points within a given distance of the query point, sorted by distance.
    pub fn within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within::<D>(query, max_dist, NonZeroUsize::MAX, true)
    }

    /// Visits every point within a given distance of the query point, unsorted.
    ///
    /// This is the lowest-overhead streaming range-query API for archived trees. It runs
    /// traversal and optimized leaf kernels, but routes each match directly to `visitor`
    /// instead of building a result collection.
    pub fn within_unsorted_visit<D, F>(&self, query: &[A; K], max_dist: D::Output, mut visitor: F)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        let mut req_ctx = ArchivedWithinUnsortedVisitReqCtx {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            let mut results = VisitorResultCollection::new(&mut visitor);
            self.process_leaf_nearest_n_within::<D, _>(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut results,
            );
        });
    }

    /// Finds all points within a given distance of the query point, unsorted.
    pub fn within_unsorted<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut results = Vec::new();
        self.within_unsorted_visit::<D, _>(query, max_dist, |result| results.push(result));
        results
    }

    /// Returns a streaming iterator over all points within a given distance, unsorted.
    ///
    /// This avoids materializing the full result set returned by [`within_unsorted`](Self::within_unsorted).
    /// For the absolute lowest overhead, use [`within_unsorted_visit`](Self::within_unsorted_visit).
    pub fn within_unsorted_iter<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> crate::kd_tree::WithinUnsortedIter<'_, Self, A, T, SS, rkyv_08::Archived<LS>, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        crate::kd_tree::WithinUnsortedIter::new(self, query, max_dist)
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + AxisUnified<Coord = A> + 'static,
    T: Basics + PartialOrd,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline(always)]
    fn process_leaf_best_n_within<D, R>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
        R: BestNeighbourResultCollection<D::Output, T>,
    {
        let threshold_item = results.threshold_item();
        match <rkyv_08::Archived<LS> as LeafStrategy<A, T, SS, K, B>>::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves().leaf_arena(leaf_idx);
                best_n_within_with_query_wide_arena::<A, T, D, R, K>(
                    &arena,
                    query_wide,
                    max_dist,
                    threshold_item,
                    results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves().leaf_view(leaf_idx);
                crate::leaf_view_chunked::best_n_within::best_n_within_with_query_wide::<
                    A,
                    T,
                    D,
                    R,
                    K,
                    B,
                >(&leaf, query_wide, max_dist, threshold_item, results);
            }
        }
    }

    /// Finds the best N points within a given distance.
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
    ) -> BinaryHeap<BestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut req_ctx = ArchivedBestNWithinReqCtx {
            query,
            max_dist,
            results: BinaryHeapResultCollection::<BestNeighbour<D::Output, T>>::with_max_qty(
                max_qty.get(),
            ),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_best_n_within::<D, _>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
        });

        req_ctx.results.into_inner()
    }
}

struct ArchivedApproxNearestOneReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    _phantom: std::marker::PhantomData<O>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ArchivedApproxNearestOneReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        panic!("approx_nearest_one should not be called with max_dist")
    }
}

struct ArchivedNearestOneReqCtx<'a, A, T, O, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    best_dist: O,
    best_item: T,
}

impl<A, T, O, const K: usize> QueryContext<A, O, K> for ArchivedNearestOneReqCtx<'_, A, T, O, K>
where
    O: AxisUnified<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.best_dist
    }

    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        true
    }
}

struct ArchivedNearestNWithinReqCtx<'a, A, T, O, R, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    _phantom: std::marker::PhantomData<T>,
}

struct ArchivedWithinUnsortedVisitReqCtx<'a, A, O, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ArchivedWithinUnsortedVisitReqCtx<'_, A, O, K>
where
    O: AxisUnified<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.max_dist
    }
}

impl<A, T, O, R, const K: usize> QueryContext<A, O, K>
    for ArchivedNearestNWithinReqCtx<'_, A, T, O, R, K>
where
    O: AxisUnified<Coord = O>,
    R: ResultCollection<O, NearestNeighbour<O, T>>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        let results_cap = self.results.threshold_distance().unwrap_or(O::max_value());
        if results_cap < self.max_dist {
            results_cap
        } else {
            self.max_dist
        }
    }
}

struct ArchivedBestNWithinReqCtx<'a, A, O, R, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
}

impl<A, O, R, const K: usize> QueryContext<A, O, K> for ArchivedBestNWithinReqCtx<'_, A, O, R, K>
where
    O: AxisUnified<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.max_dist
    }
}
