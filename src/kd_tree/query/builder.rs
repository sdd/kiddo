#![allow(private_bounds)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::hash::Hash;
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

pub(crate) trait PeriodicAxis: Axis<Coord = Self> {
    fn periodic_box_is_valid(box_size: &[Self]) -> bool {
        box_size
            .iter()
            .all(|axis_len| Self::cmp(*axis_len, Self::zero()) == Ordering::Greater)
    }
}

impl PeriodicAxis for f32 {}
impl PeriodicAxis for f64 {}
#[cfg(feature = "f16")]
impl PeriodicAxis for half::f16 {}

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

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<A, T, O: PartialOrd, const K: usize> PartialOrd for FullNearest<A, T, O, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A, T, O: PartialOrd, const K: usize> Ord for FullNearest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> PartialEq for FullBest<A, T, O, K> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}

impl<A, T: Content + PartialOrd, O: PartialEq, const K: usize> Eq for FullBest<A, T, O, K> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> PartialOrd
    for FullBest<A, T, O, K>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.item.partial_cmp(&other.item) {
            Some(Ordering::Equal) => self.distance.partial_cmp(&other.distance),
            ordering => ordering,
        }
    }
}

impl<A, T: Content + PartialOrd, O: PartialOrd, const K: usize> Ord for FullBest<A, T, O, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.item.partial_cmp(&other.item) {
            Some(Ordering::Equal) => self
                .distance
                .partial_cmp(&other.distance)
                .unwrap_or(Ordering::Equal),
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
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

/// Entry point for fluent periodic-boundary queries against a kd-tree-like accessor.
pub struct PeriodicQueryBuilder<'a, Tree, A, T, SS, LS, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
    _phantom: PhantomData<(T, SS, LS)>,
}

/// Fluent exact nearest-neighbour query.
pub struct NearestOneQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    _phantom: PhantomData<(T, SS, LS, D)>,
}

/// Fluent exact nearest-neighbour periodic query.
pub struct PeriodicNearestOneQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
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

/// Fluent nearest-N periodic query.
pub struct PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> {
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
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

/// Fluent periodic nearest-N-within query with sorted execution.
pub struct PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
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

/// Fluent periodic nearest-N-within query with unsorted execution.
pub struct PeriodicNearestNWithinUnsortedQuery<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    const K: usize,
    const B: usize,
> where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
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

/// Fluent periodic sorted within-radius query.
pub struct PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
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

/// Fluent periodic unsorted within-radius query.
pub struct PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Copy,
    D: KdTreeDistanceMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: &'a [A; K],
    radius: D::Output,
    boundary: BoundaryMode,
    _phantom: PhantomData<(T, SS, LS, D)>,
}

#[doc(hidden)]
pub struct Projected<Q, Pj> {
    inner: Q,
    _phantom: PhantomData<Pj>,
}

trait SupportsPointProjection {}

macro_rules! impl_supports_point_projection {
    ($($wrapper:ident),* $(,)?) => {
        $(
            impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize> SupportsPointProjection
                for $wrapper<'a, Tree, A, T, SS, LS, D, K, B>
            where
                A: Copy,
                D: KdTreeDistanceMetric<A, K>,
            {}
        )*
    };
}

impl_supports_point_projection!(
    NearestOneQuery,
    ApproxNearestOneQuery,
    NearestNQuery,
    NearestNUnsortedQuery,
    NearestNWithinQuery,
    NearestNWithinUnsortedQuery,
    BestNWithinQuery,
    WithinQuery,
    WithinUnsortedQuery,
);

#[allow(missing_docs)]
impl<Q, P, I, D> Projected<Q, Projection<P, I, D>>
where
    Q: SupportsPointProjection,
{
    #[inline]
    pub fn with_points(self) -> Projected<Q, Projection<Include, I, D>> {
        Projected {
            inner: self.inner,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<Q, P, I, D> Projected<Q, Projection<P, I, D>> {
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
fn assert_valid_periodic_box<A: PeriodicAxis, const K: usize>(box_size: &[A; K]) {
    assert!(
        A::periodic_box_is_valid(box_size),
        "periodic box sizes must be strictly positive"
    );
}

fn with_wrapped_queries<A: PeriodicAxis, F, const K: usize>(
    query: &[A; K],
    box_size: &[A; K],
    mut f: F,
) where
    F: FnMut(&[A; K]),
{
    fn recurse<A: PeriodicAxis, F, const K: usize>(
        query: &[A; K],
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        f: &mut F,
    ) where
        F: FnMut(&[A; K]),
    {
        if axis == K {
            f(wrapped_query);
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        wrapped_query[axis] = original - axis_len;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        wrapped_query[axis] = original;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        let mut plus = original;
        plus += axis_len;
        wrapped_query[axis] = plus;
        recurse(query, box_size, axis + 1, wrapped_query, f);

        wrapped_query[axis] = original;
    }

    let mut wrapped_query = *query;
    recurse(query, box_size, 0, &mut wrapped_query, &mut f);
}

#[inline(always)]
fn periodic_image_axis_offset<A: PeriodicAxis>(wrapped_coord: A, axis_len: A) -> A {
    if A::cmp(wrapped_coord, A::zero()) == Ordering::Less {
        A::saturating_dist(wrapped_coord, A::zero())
    } else if A::cmp(wrapped_coord, axis_len) == Ordering::Greater {
        A::saturating_dist(wrapped_coord, axis_len)
    } else {
        A::zero()
    }
}

#[inline(always)]
fn periodic_min_image_axis_delta<A: PeriodicAxis>(query: A, point: A, axis_len: A) -> A {
    let direct = A::saturating_dist(query, point);
    let wrapped = A::saturating_dist(axis_len, direct);
    if A::cmp(wrapped, direct) == Ordering::Less {
        wrapped
    } else {
        direct
    }
}

fn periodic_distance_via_minimum_image<D, A, const K: usize>(
    query: &[A; K],
    point: &[A; K],
    box_size: &[A; K],
) -> D::Output
where
    A: PeriodicAxis + 'static,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
{
    let mut distance = D::Output::zero();

    for dim in 0..K {
        let delta = periodic_min_image_axis_delta(query[dim], point[dim], box_size[dim]);
        D::combine_component(
            &mut distance,
            D::dist1(D::widen_coord(delta), D::Output::zero()),
        );
    }

    distance
}

#[derive(Clone, Copy)]
struct PeriodicImageCandidate<A, O, const K: usize> {
    wrapped_query: [A; K],
    lower_bound: O,
}

fn periodic_image_candidates<D, A, const K: usize>(
    query: &[A; K],
    box_size: &[A; K],
    threshold: Option<(BoundaryMode, D::Output)>,
    include_home: bool,
) -> Vec<PeriodicImageCandidate<A, D::Output, K>>
where
    A: PeriodicAxis + 'static,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
{
    fn recurse<D, A, const K: usize>(
        query: &[A; K],
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        has_non_zero_shift: bool,
        lower_bound: D::Output,
        threshold: Option<(BoundaryMode, D::Output)>,
        include_home: bool,
        out: &mut Vec<PeriodicImageCandidate<A, D::Output, K>>,
    ) where
        A: PeriodicAxis + 'static,
        D: KdTreeDistanceMetric<A, K>,
        D::Output: Axis<Coord = D::Output>,
    {
        if axis == K {
            if include_home || has_non_zero_shift {
                out.push(PeriodicImageCandidate {
                    wrapped_query: *wrapped_query,
                    lower_bound,
                });
            }
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        for shift in [-1_i8, 0, 1] {
            let wrapped_coord = match shift {
                -1 => original - axis_len,
                0 => original,
                1 => {
                    let mut plus = original;
                    plus += axis_len;
                    plus
                }
                _ => unreachable!(),
            };

            wrapped_query[axis] = wrapped_coord;

            let offset = periodic_image_axis_offset(wrapped_coord, axis_len);
            let mut next_lower_bound = lower_bound;
            D::combine_component(
                &mut next_lower_bound,
                D::dist1(D::widen_coord(offset), D::Output::zero()),
            );

            if threshold.is_some_and(|(boundary, limit)| {
                !boundary_accepts(boundary, next_lower_bound, limit)
            }) {
                continue;
            }

            recurse::<D, A, K>(
                query,
                box_size,
                axis + 1,
                wrapped_query,
                has_non_zero_shift || shift != 0,
                next_lower_bound,
                threshold,
                include_home,
                out,
            );
        }

        wrapped_query[axis] = original;
    }

    let mut candidates = Vec::new();
    let mut wrapped_query = *query;
    recurse::<D, A, K>(
        query,
        box_size,
        0,
        &mut wrapped_query,
        false,
        D::Output::zero(),
        threshold,
        include_home,
        &mut candidates,
    );
    candidates
}

#[inline(always)]
fn sort_periodic_image_candidates<A, O, const K: usize>(
    candidates: &mut [PeriodicImageCandidate<A, O, K>],
) where
    O: Axis<Coord = O>,
{
    candidates.sort_unstable_by(|lhs, rhs| {
        lhs.lower_bound
            .partial_cmp(&rhs.lower_bound)
            .unwrap_or(Ordering::Equal)
    });
}

#[inline(always)]
fn periodic_threshold_accepts<O: Axis<Coord = O>>(
    boundary: BoundaryMode,
    lower_bound: O,
    threshold: O,
) -> bool {
    boundary_accepts(boundary, lower_bound, threshold)
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

fn periodic_nearest_one_result<Tree, A, T, SS, LS, D, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
) -> QueryResultItem<(), T, D::Output>
where
    A: PeriodicAxis + 'static,
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
    assert_valid_periodic_box(box_size);

    if D::ORDERING != Ordering::Less {
        let mut best_result = QueryResultItem {
            point: (),
            item: T::default(),
            distance: D::Output::max_value(),
        };

        with_wrapped_queries::<A, _, K>(query, box_size, |wrapped_query| {
            let (distance, item) = tree.qb_nearest_one::<D>(wrapped_query);
            if D::Output::cmp(distance, best_result.distance) == Ordering::Less {
                best_result.distance = distance;
                best_result.item = item;
            }
        });

        return best_result;
    }

    let (home_distance, home_item) = tree.qb_nearest_one::<D>(query);
    let mut best_result = QueryResultItem {
        point: (),
        item: home_item,
        distance: home_distance,
    };

    let mut candidates = periodic_image_candidates::<D, A, K>(
        query,
        box_size,
        Some((BoundaryMode::Exclusive, best_result.distance)),
        false,
    );
    sort_periodic_image_candidates(&mut candidates);

    for candidate in candidates {
        if !periodic_threshold_accepts(
            BoundaryMode::Exclusive,
            candidate.lower_bound,
            best_result.distance,
        ) {
            continue;
        }

        let (distance, item) = tree.qb_nearest_one::<D>(&candidate.wrapped_query);
        if D::Output::cmp(distance, best_result.distance) == Ordering::Less {
            best_result.distance = distance;
            best_result.item = item;
        }
    }

    best_result
}

fn periodic_nearest_results_by_item<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<NonZeroUsize>,
    sorted: bool,
) -> Vec<QueryResultItem<(), T, D::Output>>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    assert_valid_periodic_box(box_size);

    if D::ORDERING != Ordering::Less {
        let mut best_by_item = HashMap::<T, D::Output>::new();

        with_wrapped_queries::<A, _, K>(query, box_size, |wrapped_query| {
            let candidates = match (radius, max_qty) {
                (None, Some(max_qty)) => tree.qb_nearest_n::<D>(wrapped_query, max_qty, true),
                (Some(radius), Some(max_qty)) => {
                    tree.qb_nearest_n_within::<D, EXCLUSIVE>(wrapped_query, radius, max_qty, sorted)
                }
                (Some(radius), None) if sorted => {
                    tree.qb_within::<D, EXCLUSIVE>(wrapped_query, radius)
                }
                (Some(radius), None) => {
                    tree.qb_within_unsorted::<D, EXCLUSIVE>(wrapped_query, radius)
                }
                (None, None) => {
                    unreachable!("periodic plural queries require radius and/or max_qty")
                }
            };

            for candidate in candidates {
                best_by_item
                    .entry(candidate.item)
                    .and_modify(|best_distance| {
                        if D::Output::cmp(candidate.distance, *best_distance) == Ordering::Less {
                            *best_distance = candidate.distance;
                        }
                    })
                    .or_insert(candidate.distance);
            }
        });

        let mut results: Vec<_> = best_by_item
            .into_iter()
            .map(|(item, distance)| QueryResultItem {
                point: (),
                item,
                distance,
            })
            .collect();

        if sorted {
            results.sort_unstable();
        }
        if let Some(max_qty) = max_qty {
            results.truncate(max_qty.get());
        }

        return results;
    }

    let mut best_by_item = HashMap::<T, D::Output>::new();
    let mut threshold = radius.map(|value| {
        (
            if EXCLUSIVE {
                BoundaryMode::Exclusive
            } else {
                BoundaryMode::Inclusive
            },
            value,
        )
    });

    let mut candidates = periodic_image_candidates::<D, A, K>(query, box_size, threshold, true);
    sort_periodic_image_candidates(&mut candidates);

    for candidate in candidates {
        if threshold.is_some_and(|(boundary, limit)| {
            !periodic_threshold_accepts(boundary, candidate.lower_bound, limit)
        }) {
            continue;
        }

        let candidates = match (radius, max_qty) {
            (None, Some(max_qty)) => {
                tree.qb_nearest_n::<D>(&candidate.wrapped_query, max_qty, true)
            }
            (Some(radius), Some(max_qty)) => tree.qb_nearest_n_within::<D, EXCLUSIVE>(
                &candidate.wrapped_query,
                radius,
                max_qty,
                sorted,
            ),
            (Some(radius), None) if sorted => {
                tree.qb_within::<D, EXCLUSIVE>(&candidate.wrapped_query, radius)
            }
            (Some(radius), None) => {
                tree.qb_within_unsorted::<D, EXCLUSIVE>(&candidate.wrapped_query, radius)
            }
            (None, None) => unreachable!("periodic plural queries require radius and/or max_qty"),
        };

        for periodic_candidate in candidates {
            best_by_item
                .entry(periodic_candidate.item)
                .and_modify(|best_distance| {
                    if D::Output::cmp(periodic_candidate.distance, *best_distance) == Ordering::Less
                    {
                        *best_distance = periodic_candidate.distance;
                    }
                })
                .or_insert(periodic_candidate.distance);
        }

        if radius.is_none() {
            if max_qty.is_some_and(|max_qty| best_by_item.len() >= max_qty.get()) {
                if let Some(current_worst) = best_by_item
                    .values()
                    .copied()
                    .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
                {
                    threshold = Some((BoundaryMode::Exclusive, current_worst));
                }
            }
        } else if max_qty.is_some_and(|max_qty| best_by_item.len() >= max_qty.get()) {
            if let Some(current_worst) = best_by_item
                .values()
                .copied()
                .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
            {
                let radius = radius.unwrap();
                let tightened = if D::Output::cmp(current_worst, radius) == Ordering::Less {
                    current_worst
                } else {
                    radius
                };
                threshold = Some((
                    if EXCLUSIVE {
                        BoundaryMode::Exclusive
                    } else {
                        BoundaryMode::Inclusive
                    },
                    tightened,
                ));
            }
        }
    }

    let mut results: Vec<_> = best_by_item
        .into_iter()
        .map(|(item, distance)| QueryResultItem {
            point: (),
            item,
            distance,
        })
        .collect();

    if sorted {
        results.sort_unstable();
    }
    if let Some(max_qty) = max_qty {
        results.truncate(max_qty.get());
    }

    results
}

fn scan_periodic_projected_nearest_one<Tree, A, T, SS, LS, D, Pj, const K: usize, const B: usize>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
) -> Pj::NearestOut
where
    A: PeriodicAxis + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    assert_valid_periodic_box(box_size);

    let mut best: Option<FullNearest<A, T, D::Output, K>> = None;
    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let distance = periodic_distance_via_minimum_image::<D, A, K>(query, &point, box_size);
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

fn scan_periodic_projected_nearest_results<
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    Pj,
    const K: usize,
    const B: usize,
>(
    tree: &Tree,
    query: &[A; K],
    box_size: &[A; K],
    radius: Option<D::Output>,
    max_qty: Option<usize>,
    sorted: bool,
    boundary: BoundaryMode,
) -> Vec<Pj::NearestOut>
where
    A: PeriodicAxis + 'static,
    T: Content + Copy + Eq + Hash + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: Axis<Coord = D::Output>,
    Pj: ProjectionSpec<A, T, D::Output, K>,
{
    assert_valid_periodic_box(box_size);

    let mut best_by_item = HashMap::<T, FullNearest<A, T, D::Output, K>>::new();
    for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(tree) {
        let distance = periodic_distance_via_minimum_image::<D, A, K>(query, &point, box_size);
        if radius.is_some_and(|max_dist| !boundary_accepts(boundary, distance, max_dist)) {
            continue;
        }

        let candidate = FullNearest {
            point,
            item,
            distance,
        };

        best_by_item
            .entry(item)
            .and_modify(|current| {
                if candidate < *current {
                    *current = candidate;
                }
            })
            .or_insert(candidate);
    }

    let mut results: Vec<_> = best_by_item.into_values().collect();
    if sorted {
        results.sort_unstable();
    }
    if let Some(max_qty) = max_qty {
        results.truncate(max_qty);
    }

    results
        .into_iter()
        .map(map_full_nearest::<A, T, D::Output, Pj, K>)
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
    /// Switches this query builder into periodic-boundary mode.
    #[inline]
    pub fn periodic_boundary_condition(
        self,
        box_size: &'a [A; K],
    ) -> PeriodicQueryBuilder<'a, Tree, A, T, SS, LS, K, B>
    where
        A: PeriodicAxis,
    {
        PeriodicQueryBuilder {
            tree: self.tree,
            query: self.query,
            box_size,
            _phantom: PhantomData,
        }
    }

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

impl<'a, Tree, A: Copy, T, SS, LS, const K: usize, const B: usize>
    PeriodicQueryBuilder<'a, Tree, A, T, SS, LS, K, B>
where
    A: PeriodicAxis,
{
    /// Selects an exact nearest-neighbour periodic query.
    #[inline]
    pub fn nearest_one<D>(self) -> PeriodicNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        PeriodicNearestOneQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            _phantom: PhantomData,
        }
    }

    /// Selects a nearest-N periodic query.
    #[inline]
    pub fn nearest_n<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        PeriodicNearestNQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            max_qty,
            _phantom: PhantomData,
        }
    }

    /// Selects a periodic radius query, sorted by distance by default.
    #[inline]
    pub fn within<D>(
        self,
        radius: D::Output,
    ) -> PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
    {
        PeriodicWithinQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
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
    #[inline]
    pub fn execute(self) -> QueryResultItem<(), T, D::Output> {
        periodic_nearest_one_result::<Tree, A, T, SS, LS, D, K, B>(
            self.tree,
            self.query,
            self.box_size,
        )
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn within(
        self,
        radius: D::Output,
    ) -> PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        PeriodicNearestNWithinQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            max_qty: self.max_qty,
            radius,
            boundary: BoundaryMode::Inclusive,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
            self.tree,
            self.query,
            self.box_size,
            None,
            Some(self.max_qty),
            true,
        )
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }

    #[inline]
    pub fn unsorted(self) -> PeriodicNearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        PeriodicNearestNWithinUnsortedQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            max_qty: self.max_qty,
            radius: self.radius,
            boundary: self.boundary,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    Some(self.max_qty),
                    true,
                )
            }
            BoundaryMode::Exclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, true, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    Some(self.max_qty),
                    true,
                )
            }
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicNearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    Some(self.max_qty),
                    false,
                )
            }
            BoundaryMode::Exclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, true, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    Some(self.max_qty),
                    false,
                )
            }
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }

    #[inline]
    pub fn unsorted(self) -> PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B> {
        PeriodicWithinUnsortedQuery {
            tree: self.tree,
            query: self.query,
            box_size: self.box_size,
            radius: self.radius,
            boundary: self.boundary,
            _phantom: PhantomData,
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    None,
                    true,
                )
            }
            BoundaryMode::Exclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, true, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    None,
                    true,
                )
            }
        }
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[inline]
    pub fn without_items(self) -> Projected<Self, Projection<Exclude, Exclude, Include>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn without_distances(self) -> Projected<Self, Projection<Exclude, Include, Exclude>> {
        Projected {
            inner: self,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn exclusive_boundaries(mut self) -> Self {
        self.boundary = BoundaryMode::Exclusive;
        self
    }
}

#[allow(missing_docs)]
impl<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
    PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    A: PeriodicAxis + 'static,
    T: Content + Eq + Hash + PartialOrd,
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
    #[inline]
    pub fn execute(self) -> Vec<QueryResultItem<(), T, D::Output>> {
        match self.boundary {
            BoundaryMode::Inclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    None,
                    false,
                )
            }
            BoundaryMode::Exclusive => {
                periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, true, K, B>(
                    self.tree,
                    self.query,
                    self.box_size,
                    Some(self.radius),
                    None,
                    false,
                )
            }
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

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<PeriodicNearestOneQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: PeriodicAxis + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: KdTreeDistanceMetric<A, K>,
    D::Output: crate::stem_strategy::SimdPrune
        + SimdSelectBestChildBlock3
        + BacktrackBlock3
        + BacktrackBlock4
        + TlsLeafScratch
        + 'static,
    SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    #[inline]
    pub fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_periodic_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.inner.tree,
                self.inner.query,
                self.inner.box_size,
            )
        } else {
            let result = periodic_nearest_one_result::<Tree, A, T, SS, LS, D, K, B>(
                self.inner.tree,
                self.inner.query,
                self.inner.box_size,
            );
            project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result)
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

impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: PeriodicAxis + 'static,
    T: Content + Copy + Eq + Hash + PartialOrd,
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
            scan_periodic_projected_nearest_results::<
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
                self.inner.box_size,
                None,
                Some(self.inner.max_qty.get()),
                true,
                BoundaryMode::Inclusive,
            )
        } else {
            periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                self.inner.tree,
                self.inner.query,
                self.inner.box_size,
                None,
                Some(self.inner.max_qty),
                true,
            )
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
impl_projected_boundary_methods!(PeriodicNearestNWithinQuery);
impl_projected_boundary_methods!(PeriodicNearestNWithinUnsortedQuery);
impl_projected_boundary_methods!(PeriodicWithinQuery);
impl_projected_boundary_methods!(PeriodicWithinUnsortedQuery);

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

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<PeriodicNearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn within(
        self,
        radius: D::Output,
    ) -> Projected<PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
        Projected {
            inner: self.inner.within(radius),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn unsorted(
        self,
    ) -> Projected<
        PeriodicNearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>,
        Projection<P, I, Dp>,
    > {
        Projected {
            inner: self.inner.unsorted(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
    Projected<PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
where
    A: PeriodicAxis,
    D: KdTreeDistanceMetric<A, K>,
{
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn unsorted(
        self,
    ) -> Projected<PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
    {
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

macro_rules! impl_projected_periodic_radius_execute {
    ($wrapper:ident, $sorted:expr, $max_qty:expr) => {
        impl<'a, Tree, A, T, SS, LS, D, P, I, Dp, const K: usize, const B: usize>
            Projected<$wrapper<'a, Tree, A, T, SS, LS, D, K, B>, Projection<P, I, Dp>>
        where
            A: PeriodicAxis + 'static,
            T: Content + Copy + Eq + Hash + PartialOrd,
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
                    scan_periodic_projected_nearest_results::<
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
                        self.inner.box_size,
                        Some(self.inner.radius),
                        $max_qty(&self.inner),
                        $sorted,
                        self.inner.boundary,
                    )
                } else {
                    let results = match self.inner.boundary {
                        BoundaryMode::Inclusive => {
                            periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, false, K, B>(
                                self.inner.tree,
                                self.inner.query,
                                self.inner.box_size,
                                Some(self.inner.radius),
                                $max_qty(&self.inner).map(NonZeroUsize::new).flatten(),
                                $sorted,
                            )
                        }
                        BoundaryMode::Exclusive => {
                            periodic_nearest_results_by_item::<Tree, A, T, SS, LS, D, true, K, B>(
                                self.inner.tree,
                                self.inner.query,
                                self.inner.box_size,
                                Some(self.inner.radius),
                                $max_qty(&self.inner).map(NonZeroUsize::new).flatten(),
                                $sorted,
                            )
                        }
                    };
                    results
                        .into_iter()
                        .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                        .collect()
                }
            }
        }
    };
}

impl_projected_periodic_radius_execute!(
    PeriodicNearestNWithinQuery,
    true,
    |inner: &PeriodicNearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>| Some(
        inner.max_qty.get()
    )
);
impl_projected_periodic_radius_execute!(
    PeriodicNearestNWithinUnsortedQuery,
    false,
    |inner: &PeriodicNearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>| Some(
        inner.max_qty.get()
    )
);
impl_projected_periodic_radius_execute!(
    PeriodicWithinQuery,
    true,
    |_inner: &PeriodicWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>| None
);
impl_projected_periodic_radius_execute!(
    PeriodicWithinUnsortedQuery,
    false,
    |_inner: &PeriodicWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>| None
);

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

#[cfg(test)]
mod tests {
    use crate::dist::SquaredEuclidean;
    #[cfg(feature = "rkyv_08")]
    use crate::kd_tree::ArchivedKdTree;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::FlatVec;
    #[cfg(feature = "rkyv_08")]
    use crate::leaf_strategy::VecOfArenas;
    use crate::stem_strategy::EytzingerPf;
    use std::num::NonZeroUsize;

    type WrapTree = KdTree<f64, u32, EytzingerPf<2, 8>, FlatVec<f64, u32, 2, 32>, 2, 32>;

    #[test]
    fn periodic_nearest_one_wraps_and_returns_points() {
        let points = [(1u32, [0.95, 0.5]), (2u32, [0.40, 0.5])];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let result = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        assert_eq!(result.item, 1);
        assert!((result.distance - 0.01).abs() < f64::EPSILON);

        let projected = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .without_items()
            .execute();
        assert_eq!(projected.item, ());
        assert_eq!(projected.point, ());
        assert!((projected.distance - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn periodic_radius_queries_dedup_and_respect_boundary_mode() {
        let points = [
            (1u32, [0.95, 0.5]),
            (2u32, [0.85, 0.5]),
            (3u32, [0.40, 0.5]),
        ];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];
        let boundary_radius = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute()
            .distance;

        let inclusive = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(boundary_radius)
            .unsorted()
            .execute();
        assert_eq!(inclusive.len(), 1);
        assert_eq!(inclusive[0].item, 1);

        let exclusive = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(boundary_radius)
            .exclusive_boundaries()
            .unsorted()
            .execute();
        assert!(exclusive.is_empty());

        let nearest_within = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
            .within(0.05)
            .unsorted()
            .execute();
        let mut items: Vec<_> = nearest_within
            .into_iter()
            .map(|result| result.item)
            .collect();
        items.sort_unstable();
        assert_eq!(items, vec![1, 2]);
    }

    #[cfg(feature = "rkyv_08")]
    #[test]
    fn archived_periodic_queries_match_owned() {
        type Tree = KdTree<f64, u32, EytzingerPf<2, 8>, VecOfArenas<f64, u32, 2, 32>, 2, 32>;
        type ArchivedTree =
            ArchivedKdTree<f64, u32, EytzingerPf<2, 8>, VecOfArenas<f64, u32, 2, 32>, 2, 32>;

        let points = vec![
            (1u32, [0.95, 0.5]),
            (2u32, [0.85, 0.5]),
            (3u32, [0.40, 0.5]),
        ];
        let tree = Tree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let bytes = rkyv_08::api::high::to_bytes_in::<_, rkyv_08::rancor::Error>(
            &tree,
            rkyv_08::util::AlignedVec::<128>::new(),
        )
        .unwrap();
        let archived =
            rkyv_08::access::<ArchivedTree, rkyv_08::rancor::Error>(bytes.as_slice()).unwrap();

        assert_eq!(
            archived
                .query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute(),
            tree.query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
        );
        assert_eq!(
            archived
                .query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
                .within(0.05)
                .execute(),
            tree.query(&query)
                .periodic_boundary_condition(&box_size)
                .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
                .within(0.05)
                .execute()
        );
    }
}
