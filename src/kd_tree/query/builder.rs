#![allow(private_bounds)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use std::num::NonZeroUsize;

use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
#[cfg(feature = "rkyv_08")]
use crate::kd_tree::ArchivedKdTree;
use crate::kd_tree::{KdTree, KdTreeAccessor, KdTreeIter, WithinUnsortedIter};
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, BestQueryResultItem, Content, LeafStrategy, QueryResultItem, StemStrategy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryMode {
    Inclusive,
    Exclusive,
}

#[doc(hidden)]
pub struct Include;
#[doc(hidden)]
pub struct Exclude;

#[doc(hidden)]
pub struct Projection<P, I, D>(PhantomData<(P, I, D)>);

#[doc(hidden)]
pub trait ProjectionField<T> {
    type Output;

    fn project(value: T) -> Self::Output;
}

trait PointProjectionField<A: Axis<Coord = A>, const K: usize>: ProjectionField<[A; K]> {
    fn absent_point() -> Self::Output;
}

impl<T> ProjectionField<T> for Include {
    type Output = T;

    #[inline(always)]
    fn project(value: T) -> Self::Output {
        value
    }
}

impl<T> ProjectionField<T> for Exclude {
    type Output = ();

    #[inline(always)]
    fn project(_value: T) -> Self::Output {}
}

impl<A: Axis<Coord = A>, const K: usize> PointProjectionField<A, K> for Include {
    #[inline(always)]
    fn absent_point() -> Self::Output {
        [A::zero(); K]
    }
}

impl<A: Axis<Coord = A>, const K: usize> PointProjectionField<A, K> for Exclude {
    #[inline(always)]
    fn absent_point() -> Self::Output {}
}

trait ProjectionSpec<A, T, O, const K: usize> {
    const WANTS_POINTS: bool;
    type NearestOut;
    type BestOut;

    fn nearest_from_parts(point: [A; K], item: T, distance: O) -> Self::NearestOut;
    fn best_from_parts(point: [A; K], item: T, distance: O) -> Self::BestOut;
}

impl<A, T, O, P, I, D, const K: usize> ProjectionSpec<A, T, O, K> for Projection<P, I, D>
where
    P: ProjectionField<[A; K]>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    const WANTS_POINTS: bool = std::mem::size_of::<P::Output>() != 0;
    type NearestOut = QueryResultItem<P::Output, I::Output, D::Output>;
    type BestOut = BestQueryResultItem<P::Output, I::Output, D::Output>;

    #[inline(always)]
    fn nearest_from_parts(point: [A; K], item: T, distance: O) -> Self::NearestOut {
        QueryResultItem {
            point: P::project(point),
            item: I::project(item),
            distance: D::project(distance),
        }
    }

    #[inline(always)]
    fn best_from_parts(point: [A; K], item: T, distance: O) -> Self::BestOut {
        BestQueryResultItem {
            point: P::project(point),
            item: I::project(item),
            distance: D::project(distance),
        }
    }
}

#[inline(always)]
fn project_nearest_without_point<A, T, O, P, I, D, const K: usize>(
    result: QueryResultItem<(), T, O>,
) -> QueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    QueryResultItem {
        point: P::absent_point(),
        item: I::project(result.item),
        distance: D::project(result.distance),
    }
}

#[inline(always)]
fn project_nearest_without_point_from_parts<A, T, O, P, I, D, const K: usize>(
    item: T,
    distance: O,
) -> QueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    QueryResultItem {
        point: P::absent_point(),
        item: I::project(item),
        distance: D::project(distance),
    }
}

#[inline(always)]
fn project_best_without_point<A, T, O, P, I, D, const K: usize>(
    result: BestQueryResultItem<(), T, O>,
) -> BestQueryResultItem<P::Output, I::Output, D::Output>
where
    A: Axis<Coord = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    D: ProjectionField<O>,
{
    BestQueryResultItem {
        point: P::absent_point(),
        item: I::project(result.item),
        distance: D::project(result.distance),
    }
}

#[derive(Copy, Clone)]
struct FullNearest<A, T, O, const K: usize> {
    point: [A; K],
    item: T,
    distance: O,
}

#[derive(Copy, Clone)]
struct FullBest<A, T, O, const K: usize> {
    point: [A; K],
    item: T,
    distance: O,
}

impl<A, T, O: PartialOrd, const K: usize> PartialEq for FullNearest<A, T, O, K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A, T, O: PartialOrd, const K: usize> Eq for FullNearest<A, T, O, K> {}

impl<A, T, O: PartialOrd, const K: usize> PartialOrd for FullNearest<A, T, O, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, T, O: PartialOrd, const K: usize> Ord for FullNearest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> PartialEq for FullBest<A, T, O, K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> Eq for FullBest<A, T, O, K> {}

impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> PartialOrd
    for FullBest<A, T, O, K>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> Ord for FullBest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub(crate) trait QueryBuilderTreeOps<A, T, SS, LS, const K: usize, const B: usize>:
    KdTreeAccessor<A, T, SS, LS, K, B> + Sized
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static;

    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: KdTreeDistanceMetric<A, K, Output = A>;

    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_within_unsorted<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;

    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>);

    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static;
}

impl<A, T, SS, LS, const K: usize, const B: usize> QueryBuilderTreeOps<A, T, SS, LS, K, B>
    for KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
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
        self.nearest_one::<D>(query)
    }

    #[inline]
    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: KdTreeDistanceMetric<A, K, Output = A>,
    {
        self.approx_nearest_one::<D>(query)
    }

    #[inline]
    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n::<D>(query, max_qty, sorted)
    }

    #[inline]
    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty, sorted)
    }

    #[inline]
    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_impl::<D, EXCLUSIVE>(query, radius)
    }

    #[inline]
    fn qb_within_unsorted<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_unsorted_impl::<D, EXCLUSIVE>(query, radius)
    }

    #[inline]
    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock4
            + BacktrackBlock3
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.best_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty)
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize>
    QueryBuilderTreeOps<A, T, SS, rkyv_08::Archived<LS>, K, B>
    for ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn qb_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
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
        self.nearest_one::<D>(query)
    }

    #[inline]
    fn qb_approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        T: Copy + Default + PartialOrd + PartialEq,
        D: KdTreeDistanceMetric<A, K, Output = A>,
    {
        self.approx_nearest_one::<D>(query)
    }

    #[inline]
    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n::<D>(query, max_qty, sorted)
    }

    #[inline]
    fn qb_nearest_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty, sorted)
    }

    #[inline]
    fn qb_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_impl::<D, EXCLUSIVE>(query, radius)
    }

    #[inline]
    fn qb_within_unsorted<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.within_unsorted_impl::<D, EXCLUSIVE>(query, radius)
    }

    #[inline]
    fn qb_within_unsorted_visit<D, F, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        visitor: F,
    ) where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.best_n_within_impl::<D, EXCLUSIVE>(query, radius, max_qty)
    }
}

/// Entry point for fluent queries against a kd-tree-like accessor.
pub struct QueryBuilder<'a, Tree, A, T, SS, LS, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    _phantom: PhantomData<(T, SS, LS)>,
}

/// Fluent exact nearest-neighbour query.
pub struct NearestOneQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent approximate nearest-neighbour query.
pub struct ApproxNearestOneQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent nearest-N query with optional unsorted execution.
pub struct NearestNQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    max_qty: NonZeroUsize,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent unsorted nearest-N query.
pub struct NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    max_qty: NonZeroUsize,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent best-N-within query.
pub struct BestNWithinQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    max_qty: NonZero<usize>,
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent nearest-N-within query with sorted execution.
pub struct NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    max_qty: NonZeroUsize,
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent nearest-N-within query with unsorted execution.
pub struct NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    max_qty: NonZeroUsize,
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent sorted within-radius query.
pub struct WithinQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent unsorted within-radius query.
pub struct WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

#[doc(hidden)]
pub struct Projected<Q, Pj> {
    inner: Q,
    _phantom: PhantomData<Pj>,
}

#[allow(missing_docs)]
impl<Q, P, I, D> Projected<Q, Projection<P, I, D>> {
    #[inline]
    pub fn with_points(self) -> Projected<Q, Projection<Include, I, D>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_points(self) -> Projected<Q, Projection<Exclude, I, D>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn with_items(self) -> Projected<Q, Projection<P, Include, D>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_items(self) -> Projected<Q, Projection<P, Exclude, D>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn with_distances(self) -> Projected<Q, Projection<P, I, Include>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Q, Projection<P, I, Exclude>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }
}

enum WithinUnsortedBuilderIter<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    Inclusive(WithinUnsortedIter<'a, Tree, A, T, SS, LS, D, false, K, B>),
    Exclusive(WithinUnsortedIter<'a, Tree, A, T, SS, LS, D, true, K, B>),
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> Iterator
    for WithinUnsortedBuilderIter<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    type Item = QueryResultItem<(), T, D::Output>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inclusive(iter) => iter.next(),
            Self::Exclusive(iter) => iter.next(),
        }
    }
}

#[inline(always)]
fn boundary_accepts<O: Axis<Coord = O>>(boundary: BoundaryMode, distance: O, radius: O) -> bool {
    match boundary {
        BoundaryMode::Inclusive => O::cmp(distance, radius) != Ordering::Greater,
        BoundaryMode::Exclusive => O::cmp(distance, radius) == Ordering::Less,
    }
}

#[inline(always)]
fn map_full_nearest<A, T, O, Pj, const K: usize>(
    candidate: FullNearest<A, T, O, K>,
) -> Pj::NearestOut
where
    Pj: ProjectionSpec<A, T, O, K>,
{
    Pj::nearest_from_parts(candidate.point, candidate.item, candidate.distance)
}

#[inline(always)]
fn map_full_best<A, T, O, Pj, const K: usize>(candidate: FullBest<A, T, O, K>) -> Pj::BestOut
where
    Pj: ProjectionSpec<A, T, O, K>,
{
    Pj::best_from_parts(candidate.point, candidate.item, candidate.distance)
}

fn scan_projected_nearest_one<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
) -> Pj::NearestOut
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    let query_wide = query.map(D::widen_coord);
    let mut best: Option<FullNearest<A, T, D::Output, K>> = None;

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);

        let candidate = FullNearest {
            point,
            item,
            distance,
        };

        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
        }
    }

    let best = best.unwrap_or(FullNearest {
        point: [A::zero(); K],
        item: T::default(),
        distance: D::Output::max_value(),
    });

    map_full_nearest::<A, T, D::Output, Pj, K>(best)
}

fn scan_projected_nearest_results<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<usize>,
    sorted: bool,
    boundary: BoundaryMode,
) -> Vec<Pj::NearestOut>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    let query_wide = query.map(D::widen_coord);
    let mut results = Vec::new();

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);

        if radius.is_none_or(|max_dist| boundary_accepts(boundary, distance, max_dist)) {
            results.push(FullNearest {
                point,
                item,
                distance,
            });
        }
    }

    results.sort_unstable();

    if let Some(max_qty) = max_qty {
        results.truncate(max_qty);
    }

    if !sorted {
        // Preserve current API contract only loosely for projected point-carrying fallbacks:
        // order is unspecified for unsorted queries.
    }

    results
        .into_iter()
        .map(map_full_nearest::<A, T, D::Output, Pj, K>)
        .collect()
}

fn scan_projected_best_results<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    radius: D::Output,
    max_qty: usize,
    boundary: BoundaryMode,
) -> BinaryHeap<Pj::BestOut>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
    Pj::BestOut: Ord,
{
    let query_wide = query.map(D::widen_coord);
    let mut heap = BinaryHeap::<FullBest<A, T, D::Output, K>>::with_capacity(max_qty);

    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let point_wide = point.map(D::widen_coord);
        let distance = D::dist(&point_wide, &query_wide);

        if !boundary_accepts(boundary, distance, radius) {
            continue;
        }

        let candidate = FullBest {
            point,
            item,
            distance,
        };

        if heap.len() < max_qty {
            heap.push(candidate);
        } else if candidate < *heap.peek().unwrap() {
            *heap.peek_mut().unwrap() = candidate;
        }
    }

    heap.into_iter()
        .map(map_full_best::<A, T, D::Output, Pj, K>)
        .collect()
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B> {
    /// Starts a fluent query against this tree.
    #[inline]
    pub fn query<'a>(&'a self, query: &'a [A; K]) -> QueryBuilder<'a, Self, A, T, SS, LS, K, B> {
        QueryBuilder {
            tree: self,
            query,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SS, LS, const K: usize, const B: usize> ArchivedKdTree<A, T, SS, LS, K, B>
where
    A: rkyv_08::Archive + Axis<Coord = A>,
    T: Content,
    SS: StemStrategy,
    LS: rkyv_08::Archive,
    rkyv_08::Archived<LS>: LeafStrategy<A, T, SS, K, B>,
{
    /// Starts a fluent query against this archived tree.
    #[inline]
    pub fn query<'a>(
        &'a self,
        query: &'a [A; K],
    ) -> QueryBuilder<'a, Self, A, T, SS, rkyv_08::Archived<LS>, K, B> {
        QueryBuilder {
            tree: self,
            query,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, const K: usize, const B: usize>
    QueryBuilder<'a, Tree, A, T, SS, LS, K, B>
{
    /// Selects an exact nearest-neighbour query.
    #[inline]
    pub fn nearest_one<D>(self) -> NearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        NearestOneQuery {
            tree: self.tree,
            query: self.query,
            _phantom: PhantomData,
        }
    }

    /// Selects a nearest-N query, sorted by distance by default.
    #[inline]
    pub fn nearest_n<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        NearestNQuery {
            tree: self.tree,
            query: self.query,
            max_qty,
            _phantom: PhantomData,
        }
    }

    /// Selects a radius query, sorted by distance by default.
    #[inline]
    pub fn within<D>(self, radius: D::Output) -> WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        WithinQuery {
            tree: self.tree,
            query: self.query,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }

    /// Selects a best-N-within query.
    #[inline]
    pub fn best_n_within<D>(
        self,
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        T: Content + PartialOrd + 'static,
        D: KdTreeDistanceMetric<A, K>,
    {
        BestNWithinQuery {
            tree: self.tree,
            query: self.query,
            max_qty,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Switches to approximate nearest-neighbour search.
    #[inline]
    pub fn approx(self) -> ApproxNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        ApproxNearestOneQuery {
            tree: self.tree,
            query: self.query,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    NearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
{
    /// Executes the exact nearest-neighbour query.
    #[inline]
    pub fn execute(self) -> QueryResultItem<(), T, D::Output> {
        let (distance, item) = self.tree.qb_nearest_one::<D>(self.query);
        project_nearest_without_point_from_parts::<A, T, D::Output, Exclude, Include, Include, K>(
            item, distance,
        )
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    ApproxNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K, Output = A>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Executes the approximate nearest-neighbour query.
    #[inline]
    pub fn execute(self) -> QueryResultItem<(), T, D::Output> {
        let (distance, item) = self.tree.qb_approx_nearest_one::<D>(self.query);
        project_nearest_without_point_from_parts::<A, T, D::Output, Exclude, Include, Include, K>(
            item, distance,
        )
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Adds a radius bound to the nearest-N query.
    #[inline]
    pub fn within(self, radius: D::Output) -> NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        NearestNWithinQuery {
            tree: self.tree,
            query: self.query,
            max_qty: self.max_qty,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }

    /// Executes the nearest-N query without sorting.
    #[inline]
    pub fn unsorted(self) -> NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        NearestNUnsortedQuery {
            tree: self.tree,
            query: self.query,
            max_qty: self.max_qty,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the nearest-N query sorted by distance.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        self.tree
            .qb_nearest_n::<D>(self.query, self.max_qty, true)
            .into_iter()
            .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
            .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Adds a radius bound to the unsorted nearest-N query.
    #[inline]
    pub fn within(
        self,
        radius: D::Output,
    ) -> NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        NearestNWithinUnsortedQuery {
            tree: self.tree,
            query: self.query,
            max_qty: self.max_qty,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the nearest-N query without sorting.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        self.tree
            .qb_nearest_n::<D>(self.query, self.max_qty, false)
            .into_iter()
            .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
            .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Excludes points lying exactly on the radius boundary.
    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }

    /// Executes the nearest-N-within query without sorting.
    #[inline]
    pub fn unsorted(self) -> NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        NearestNWithinUnsortedQuery {
            tree: self.tree,
            query: self.query,
            max_qty: self.max_qty,
            radius: self.radius,
            boundary: self.boundary,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the nearest-N-within query sorted by distance.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => self.tree.qb_nearest_n_within::<D, false>(
                self.query,
                self.radius,
                self.max_qty,
                true,
            ),
            BoundaryMode::Exclusive => self.tree.qb_nearest_n_within::<D, true>(
                self.query,
                self.radius,
                self.max_qty,
                true,
            ),
        }
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
        .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Excludes points lying exactly on the radius boundary.
    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the nearest-N-within query without sorting.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => self.tree.qb_nearest_n_within::<D, false>(
                self.query,
                self.radius,
                self.max_qty,
                false,
            ),
            BoundaryMode::Exclusive => self.tree.qb_nearest_n_within::<D, true>(
                self.query,
                self.radius,
                self.max_qty,
                false,
            ),
        }
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
        .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    BestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Excludes points lying exactly on the radius boundary.
    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    BestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the best-N-within query.
    #[inline]
    pub fn execute(self) -> BinaryHeap<BestQueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => {
                self.tree
                    .qb_best_n_within::<D, false>(self.query, self.radius, self.max_qty)
            }
            BoundaryMode::Exclusive => {
                self.tree
                    .qb_best_n_within::<D, true>(self.query, self.radius, self.max_qty)
            }
        }
        .into_iter()
        .map(project_best_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
        .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Excludes points lying exactly on the radius boundary.
    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }

    /// Switches to unsorted within-radius execution.
    #[inline]
    pub fn unsorted(self) -> WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        WithinUnsortedQuery {
            tree: self.tree,
            query: self.query,
            radius: self.radius,
            boundary: self.boundary,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the within-radius query sorted by distance.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => self.tree.qb_within::<D, false>(self.query, self.radius),
            BoundaryMode::Exclusive => self.tree.qb_within::<D, true>(self.query, self.radius),
        }
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
        .collect()
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
    /// Includes point coordinates in the returned result.
    #[inline]
    pub fn with_points(self) -> Projected<Self, Projection<Include, Include, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits items from the returned result.
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Omits distances from the returned result.
    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// Excludes points lying exactly on the radius boundary.
    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }
}

impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
{
    /// Executes the unsorted within-radius query.
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => self
                .tree
                .qb_within_unsorted::<D, false>(self.query, self.radius),
            BoundaryMode::Exclusive => self
                .tree
                .qb_within_unsorted::<D, true>(self.query, self.radius),
        }
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
        .collect()
    }

    /// Streams unsorted within-radius results directly to a visitor.
    #[inline]
    pub fn visit<F>(self, visitor: F)
    where
        F: FnMut(QueryResultItem<(), T, D::Output>),
    {
        match self.boundary {
            BoundaryMode::Inclusive => {
                let mut visitor = visitor;
                self.tree.qb_within_unsorted_visit::<D, _, false>(
                    self.query,
                    self.radius,
                    move |result| {
                        visitor(project_nearest_without_point::<
                            A,
                            T,
                            D::Output,
                            Exclude,
                            Include,
                            Include,
                            K,
                        >(result))
                    },
                )
            }
            BoundaryMode::Exclusive => {
                let mut visitor = visitor;
                self.tree.qb_within_unsorted_visit::<D, _, true>(
                    self.query,
                    self.radius,
                    move |result| {
                        visitor(project_nearest_without_point::<
                            A,
                            T,
                            D::Output,
                            Exclude,
                            Include,
                            Include,
                            K,
                        >(result))
                    },
                )
            }
        }
    }

    /// Returns an iterator over unsorted within-radius results.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = QueryResultItem<(), T, D::Output>> + 'a {
        match self.boundary {
            BoundaryMode::Inclusive => {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Inclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, false, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            }
            BoundaryMode::Exclusive => {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Exclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, true, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            }
        }
        .map(project_nearest_without_point::<A, T, D::Output, Exclude, Include, Include, K>)
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn approx(
        self,
    ) -> Projected<ApproxNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
        Projected {
            inner: self.inner.approx(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
{
    #[inline]
    pub fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output>
    where
        P: PointProjectionField<A, K>,
        I: ProjectionField<T>,
        Dp: ProjectionField<D::Output>,
    {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
            )
        } else {
            let (distance, item) = self.inner.tree.qb_nearest_one::<D>(self.inner.query);
            project_nearest_without_point_from_parts::<A, T, D::Output, P, I, Dp, K>(item, distance)
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<ApproxNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K, Output = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    #[inline]
    pub fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
            )
        } else {
            let (distance, item) = self.inner.tree.qb_approx_nearest_one::<D>(self.inner.query);
            project_nearest_without_point_from_parts::<A, T, D::Output, P, I, Dp, K>(item, distance)
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn within(
        self,
        radius: D::Output,
    ) -> Projected<NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>> {
        Projected {
            inner: self.inner.within(radius),
            _phantom: PhantomData,
        }
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn unsorted(
        self,
    ) -> Projected<NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
        Projected {
            inner: self.inner.unsorted(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                None,
                Some(self.inner.max_qty.get()),
                true,
                BoundaryMode::Inclusive,
            )
        } else {
            self.inner
                .tree
                .qb_nearest_n::<D>(self.inner.query, self.inner.max_qty, true)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn within(
        self,
        radius: D::Output,
    ) -> Projected<NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
        Projected {
            inner: self.inner.within(radius),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                None,
                Some(self.inner.max_qty.get()),
                false,
                BoundaryMode::Inclusive,
            )
        } else {
            self.inner
                .tree
                .qb_nearest_n::<D>(self.inner.query, self.inner.max_qty, false)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

macro_rules! impl_projected_boundary_methods {
    ($wrapper:ident) => {
        impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
            Projected<$wrapper<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
        where
            D: KdTreeDistanceMetric<A, K>,
        {
            #[inline]
            pub fn exclusive_boundaries(mut self) -> Self {
                self.inner.boundary = BoundaryMode::Exclusive;
                self
            }
        }
    };
}

impl_projected_boundary_methods!(NearestNWithinQuery);
impl_projected_boundary_methods!(NearestNWithinUnsortedQuery);
impl_projected_boundary_methods!(WithinQuery);
impl_projected_boundary_methods!(WithinUnsortedQuery);
impl_projected_boundary_methods!(BestNWithinQuery);

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn unsorted(
        self,
    ) -> Projected<NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
        Projected {
            inner: self.inner.unsorted(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn unsorted(
        self,
    ) -> Projected<WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>> {
        Projected {
            inner: self.inner.unsorted(),
            _phantom: PhantomData,
        }
    }
}

macro_rules! impl_projected_radius_execute {
    ($wrapper:ident, $sorted:expr, $field:ident, $tree_fn:ident) => {
        impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
            Projected<$wrapper<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
        where
            A: Axis<Coord = A> + 'static,
            T: Content + Copy + PartialOrd,
            SS: StemStrategy,
            LS: LeafStrategy<A, T, SS, K, B>,
            Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
            D: KdTreeDistanceMetric<A, K>,
            D::Output: crate::stem_strategy::SimdPrune
                + SimdSelectBestChildBlock3
                + BacktrackBlock3
                + BacktrackBlock4
                + TlsLeafScratch
                + 'static,
            SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
            P: PointProjectionField<A, K>,
            I: ProjectionField<T>,
            Dp: ProjectionField<D::Output>,
        {
            #[inline]
            pub fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
                if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
                    scan_projected_nearest_results::<
                        Tree,
                        A,
                        T,
                        SS,
                        LS,
                        D,
                        Projection<P, I, Dp>,
                        K,
                        B,
                    >(
                        self.inner.tree,
                        self.inner.query,
                        Some(self.inner.radius),
                        Some(self.inner.max_qty.get()),
                        $sorted,
                        self.inner.boundary,
                    )
                } else {
                    match self.inner.boundary {
                        BoundaryMode::Inclusive => self.inner.tree.$tree_fn::<D, false>(
                            self.inner.query,
                            self.inner.radius,
                            self.inner.max_qty,
                            $sorted,
                        ),
                        BoundaryMode::Exclusive => self.inner.tree.$tree_fn::<D, true>(
                            self.inner.query,
                            self.inner.radius,
                            self.inner.max_qty,
                            $sorted,
                        ),
                    }
                    .into_iter()
                    .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                    .collect()
                }
            }
        }
    };
}

impl_projected_radius_execute!(NearestNWithinQuery, true, max_qty, qb_nearest_n_within);
impl_projected_radius_execute!(
    NearestNWithinUnsortedQuery,
    false,
    max_qty,
    qb_nearest_n_within
);

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                Some(self.inner.radius),
                None,
                true,
                self.inner.boundary,
            )
        } else {
            match self.inner.boundary {
                BoundaryMode::Inclusive => self
                    .inner
                    .tree
                    .qb_within::<D, false>(self.inner.query, self.inner.radius),
                BoundaryMode::Exclusive => self
                    .inner
                    .tree
                    .qb_within::<D, true>(self.inner.query, self.inner.radius),
            }
            .into_iter()
            .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
            .collect()
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B> + 'a,
    D: KdTreeDistanceMetric<A, K> + 'a,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K> + 'a,
    I: ProjectionField<T> + 'a,
    Dp: ProjectionField<D::Output> + 'a,
{
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                Some(self.inner.radius),
                None,
                false,
                self.inner.boundary,
            )
        } else {
            match self.inner.boundary {
                BoundaryMode::Inclusive => self
                    .inner
                    .tree
                    .qb_within_unsorted::<D, false>(self.inner.query, self.inner.radius),
                BoundaryMode::Exclusive => self
                    .inner
                    .tree
                    .qb_within_unsorted::<D, true>(self.inner.query, self.inner.radius),
            }
            .into_iter()
            .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
            .collect()
        }
    }

    #[inline]
    pub fn visit<F>(self, mut visitor: F)
    where
        F: FnMut(QueryResultItem<P::Output, I::Output, Dp::Output>),
    {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            let query_wide = self.inner.query.map(D::widen_coord);
            for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(self.inner.tree) {
                let point_wide = point.map(D::widen_coord);
                let distance = D::dist(&point_wide, &query_wide);
                if boundary_accepts(self.inner.boundary, distance, self.inner.radius) {
                    visitor(Projection::<P, I, Dp>::nearest_from_parts(
                        point, item, distance,
                    ));
                }
            }
        } else {
            match self.inner.boundary {
                BoundaryMode::Inclusive => self.inner.tree.qb_within_unsorted_visit::<D, _, false>(
                    self.inner.query,
                    self.inner.radius,
                    move |result| {
                        visitor(
                            project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result),
                        )
                    },
                ),
                BoundaryMode::Exclusive => self.inner.tree.qb_within_unsorted_visit::<D, _, true>(
                    self.inner.query,
                    self.inner.radius,
                    move |result| {
                        visitor(
                            project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result),
                        )
                    },
                ),
            }
        }
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn iter(
        self,
    ) -> Box<dyn Iterator<Item = QueryResultItem<P::Output, I::Output, Dp::Output>> + 'a> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            let query_wide = self.inner.query.map(D::widen_coord);
            let radius = self.inner.radius;
            let boundary = self.inner.boundary;
            Box::new(
                KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(self.inner.tree).filter_map(
                    move |(item, point)| {
                        let point_wide = point.map(D::widen_coord);
                        let distance = D::dist(&point_wide, &query_wide);
                        boundary_accepts(boundary, distance, radius).then(|| {
                            Projection::<P, I, Dp>::nearest_from_parts(point, item, distance)
                        })
                    },
                ),
            )
        } else {
            let iter = match self.inner.boundary {
                BoundaryMode::Inclusive => {
                    WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Inclusive(
                        WithinUnsortedIter::<_, _, _, _, _, _, false, K, B>::new(
                            self.inner.tree,
                            self.inner.query,
                            self.inner.radius,
                        ),
                    )
                }
                BoundaryMode::Exclusive => {
                    WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Exclusive(
                        WithinUnsortedIter::<_, _, _, _, _, _, true, K, B>::new(
                            self.inner.tree,
                            self.inner.query,
                            self.inner.radius,
                        ),
                    )
                }
            };
            Box::new(iter.map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>))
        }
    }
}

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<BestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
    BestQueryResultItem<P::Output, I::Output, Dp::Output>: Ord,
{
    #[inline]
    pub fn execute(self) -> BinaryHeap<BestQueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_best_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                self.inner.radius,
                self.inner.max_qty.get(),
                self.inner.boundary,
            )
        } else {
            match self.inner.boundary {
                BoundaryMode::Inclusive => self.inner.tree.qb_best_n_within::<D, false>(
                    self.inner.query,
                    self.inner.radius,
                    self.inner.max_qty,
                ),
                BoundaryMode::Exclusive => self.inner.tree.qb_best_n_within::<D, true>(
                    self.inner.query,
                    self.inner.radius,
                    self.inner.max_qty,
                ),
            }
            .into_iter()
            .map(project_best_without_point::<A, T, D::Output, P, I, Dp, K>)
            .collect()
        }
    }
}
