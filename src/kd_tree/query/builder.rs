#![allow(private_bounds)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::{NonZero, NonZeroUsize};

use crate::dist::{DistanceMetricCore, DistanceMetricSimdBlock};
use crate::kd_tree::query_stack::StackTrait;
#[cfg(feature = "rkyv_08")]
use crate::kd_tree::ArchivedKdTree;
use crate::kd_tree::{KdTree, KdTreeAccessor, KdTreeIter, KdTreeQueryOps, WithinUnsortedIter};
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, BestQueryResultItem, Content, LeafStrategy, QueryResultItem, StemStrategy};

use super::periodic::{periodic_nearest_one_result, periodic_nearest_results};

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

pub(crate) trait ProjectionSpec<A, T, O, const K: usize> {
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
pub(crate) struct FullNearest<A, T, O, const K: usize> {
    pub(crate) point: [A; K],
    pub(crate) item: T,
    pub(crate) distance: O,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K, Output = A>;

    fn qb_nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
    where
        T: PartialOrd,
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K, Output = A>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K, Output = A>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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
        D: DistanceMetricSimdBlock<A, K>,
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

#[doc(hidden)]
pub struct NoMetric;

pub struct SelectedMetric<D>(PhantomData<D>);
#[doc(hidden)]
pub struct RootState;
#[doc(hidden)]
pub struct CartesianSpace;
#[doc(hidden)]
pub struct PeriodicSpace;
#[doc(hidden)]
pub struct NearestOneState;
#[doc(hidden)]
pub struct ApproxNearestOneState;
#[doc(hidden)]
pub struct NearestNState;
#[doc(hidden)]
pub struct NearestNWithinState;
#[doc(hidden)]
pub struct WithinState;
#[doc(hidden)]
pub struct BestNWithinState;

trait ResultFamily {}
impl ResultFamily for NearestOneState {}
impl ResultFamily for ApproxNearestOneState {}
impl ResultFamily for NearestNState {}
impl ResultFamily for NearestNWithinState {}
impl ResultFamily for WithinState {}
impl ResultFamily for BestNWithinState {}

trait BuilderMetric<A, const K: usize> {
    type Output: Copy;
}

impl<A, const K: usize> BuilderMetric<A, K> for NoMetric {
    type Output = ();
}

impl<A: Copy, D, const K: usize> BuilderMetric<A, K> for SelectedMetric<D>
where
    D: DistanceMetricCore<A>,
{
    type Output = D::Output;
}

/// A fluent query builder created by [`KdTree::query`](crate::KdTree::query).
///
/// Chain one of the query selectors such as [`nearest_one`](Self::nearest_one),
/// [`nearest_n`](Self::nearest_n), [`within`](Self::within), or
/// [`best_n_within`](Self::best_n_within), optionally refine the query, and
/// then call a terminal method such as [`execute`](Self::execute).
///
/// The concrete generic parameters on this type encode the current builder
/// state and are considered an implementation detail of the fluent API.
pub struct QueryBuilder<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    D,
    Family,
    Space,
    Pj,
    const SORTED: bool,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
> where
    D: BuilderMetric<A, K>,
{
    tree: &'a Tree,
    query: &'a [A; K],
    box_size: Option<&'a [A; K]>,
    max_qty: Option<NonZeroUsize>,
    radius: <D as BuilderMetric<A, K>>::Output,
    _phantom: PhantomData<(T, SS, LS, Family, Space, Pj)>,
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > Copy for QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > Clone for QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

type RootQueryBuilder<'a, Tree, A, T, SS, LS, const K: usize, const B: usize> = QueryBuilder<
    'a,
    Tree,
    A,
    T,
    SS,
    LS,
    NoMetric,
    RootState,
    CartesianSpace,
    Projection<Exclude, Include, Include>,
    true,
    false,
    K,
    B,
>;

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    #[inline(always)]
    fn rebind<D2, Family2, Space2, Pj2, const SORTED2: bool, const EXCLUSIVE2: bool>(
        self,
        box_size: Option<&'a [A; K]>,
        max_qty: Option<NonZeroUsize>,
        radius: <D2 as BuilderMetric<A, K>>::Output,
    ) -> QueryBuilder<'a, Tree, A, T, SS, LS, D2, Family2, Space2, Pj2, SORTED2, EXCLUSIVE2, K, B>
    where
        D2: BuilderMetric<A, K>,
    {
        QueryBuilder {
            tree: self.tree,
            query: self.query,
            box_size,
            max_qty,
            radius,
            _phantom: PhantomData,
        }
    }
}

#[doc(hidden)]
pub trait ExecuteQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn execute(self) -> Self::Output;
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
    Self: ExecuteQueryBuilder,
{
    /// Executes the configured query and returns the query-specific result type.
    #[inline]
    pub fn execute(self) -> <Self as ExecuteQueryBuilder>::Output {
        <Self as ExecuteQueryBuilder>::execute(self)
    }
}

#[doc(hidden)]
pub trait WithoutPointsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_points(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithItemsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_items(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithoutItemsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_items(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithDistancesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_distances(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithoutDistancesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn without_distances(self) -> Self::Output;
}

#[doc(hidden)]
pub trait WithPointsQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn with_points(self) -> Self::Output;
}

#[doc(hidden)]
pub trait PeriodicBoundaryConditionQueryBuilder<'a, A, const K: usize> {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn periodic_boundary_condition(self, box_size: &'a [A; K]) -> Self::Output;
}

#[doc(hidden)]
pub trait NearestOneQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetricSimdBlock<A, K>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn nearest_one(self) -> Self::Output;
}

#[doc(hidden)]
pub trait NearestNQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetricSimdBlock<A, K>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn nearest_n(self, max_qty: NonZeroUsize) -> Self::Output;
}

#[doc(hidden)]
pub trait WithinQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetricSimdBlock<A, K>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn within(self, radius: D::Output) -> Self::Output;
}

#[doc(hidden)]
pub trait BestNWithinQueryBuilder<A: Copy, const K: usize, D>
where
    D: DistanceMetricSimdBlock<A, K>,
{
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn best_n_within(self, radius: D::Output, max_qty: NonZero<usize>) -> Self::Output;
}

#[doc(hidden)]
pub trait ApproxQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn approx(self) -> Self::Output;
}

#[doc(hidden)]
pub trait UnsortedQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn unsorted(self) -> Self::Output;
}

#[doc(hidden)]
pub trait ExclusiveBoundariesQueryBuilder {
    #[doc(hidden)]
    type Output;

    #[doc(hidden)]
    fn exclusive_boundaries(self) -> Self::Output;
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > QueryBuilder<'a, Tree, A, T, SS, LS, D, Family, Space, Pj, SORTED, EXCLUSIVE, K, B>
where
    D: BuilderMetric<A, K>,
{
    /// Excludes stored point coordinates from the eventual query results.
    #[inline]
    pub fn without_points(self) -> <Self as WithoutPointsQueryBuilder>::Output
    where
        Self: WithoutPointsQueryBuilder,
    {
        <Self as WithoutPointsQueryBuilder>::without_points(self)
    }

    /// Includes stored items in the eventual query results.
    #[inline]
    pub fn with_items(self) -> <Self as WithItemsQueryBuilder>::Output
    where
        Self: WithItemsQueryBuilder,
    {
        <Self as WithItemsQueryBuilder>::with_items(self)
    }

    /// Excludes stored items from the eventual query results.
    #[inline]
    pub fn without_items(self) -> <Self as WithoutItemsQueryBuilder>::Output
    where
        Self: WithoutItemsQueryBuilder,
    {
        <Self as WithoutItemsQueryBuilder>::without_items(self)
    }

    /// Includes distances in the eventual query results.
    #[inline]
    pub fn with_distances(self) -> <Self as WithDistancesQueryBuilder>::Output
    where
        Self: WithDistancesQueryBuilder,
    {
        <Self as WithDistancesQueryBuilder>::with_distances(self)
    }

    /// Excludes distances from the eventual query results.
    #[inline]
    pub fn without_distances(self) -> <Self as WithoutDistancesQueryBuilder>::Output
    where
        Self: WithoutDistancesQueryBuilder,
    {
        <Self as WithoutDistancesQueryBuilder>::without_distances(self)
    }

    /// Includes stored point coordinates in the eventual query results.
    ///
    /// This is available only for non-periodic queries.
    #[inline]
    pub fn with_points(self) -> <Self as WithPointsQueryBuilder>::Output
    where
        Self: WithPointsQueryBuilder,
    {
        <Self as WithPointsQueryBuilder>::with_points(self)
    }

    /// Interprets subsequent queries using periodic boundary conditions.
    ///
    /// `box_size` gives the period along each axis and every value must be
    /// strictly positive.
    #[inline]
    pub fn periodic_boundary_condition(
        self,
        box_size: &'a [A; K],
    ) -> <Self as PeriodicBoundaryConditionQueryBuilder<'a, A, K>>::Output
    where
        Self: PeriodicBoundaryConditionQueryBuilder<'a, A, K>,
    {
        <Self as PeriodicBoundaryConditionQueryBuilder<'a, A, K>>::periodic_boundary_condition(
            self, box_size,
        )
    }

    /// Selects an exact nearest-neighbour query.
    #[inline]
    pub fn nearest_one<Dq>(self) -> <Self as NearestOneQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetricSimdBlock<A, K>,
        Self: NearestOneQueryBuilder<A, K, Dq>,
    {
        <Self as NearestOneQueryBuilder<A, K, Dq>>::nearest_one(self)
    }

    /// Selects a k-nearest-neighbours query.
    #[inline]
    pub fn nearest_n<Dq>(
        self,
        max_qty: NonZeroUsize,
    ) -> <Self as NearestNQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetricSimdBlock<A, K>,
        Self: NearestNQueryBuilder<A, K, Dq>,
    {
        <Self as NearestNQueryBuilder<A, K, Dq>>::nearest_n(self, max_qty)
    }

    /// Selects a radius query or adds a radius bound to a k-nearest-neighbours query.
    #[inline]
    pub fn within<Dq>(self, radius: Dq::Output) -> <Self as WithinQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetricSimdBlock<A, K>,
        Self: WithinQueryBuilder<A, K, Dq>,
    {
        <Self as WithinQueryBuilder<A, K, Dq>>::within(self, radius)
    }

    /// Selects a radius query that keeps the best `max_qty` entries by item ordering.
    #[inline]
    pub fn best_n_within<Dq>(
        self,
        radius: Dq::Output,
        max_qty: NonZero<usize>,
    ) -> <Self as BestNWithinQueryBuilder<A, K, Dq>>::Output
    where
        A: Copy,
        Dq: DistanceMetricSimdBlock<A, K>,
        Self: BestNWithinQueryBuilder<A, K, Dq>,
    {
        <Self as BestNWithinQueryBuilder<A, K, Dq>>::best_n_within(self, radius, max_qty)
    }

    /// Switches exact nearest-neighbour search to approximate descent-only mode.
    #[inline]
    pub fn approx(self) -> <Self as ApproxQueryBuilder>::Output
    where
        Self: ApproxQueryBuilder,
    {
        <Self as ApproxQueryBuilder>::approx(self)
    }

    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    pub fn unsorted(self) -> <Self as UnsortedQueryBuilder>::Output
    where
        Self: UnsortedQueryBuilder,
    {
        <Self as UnsortedQueryBuilder>::unsorted(self)
    }

    /// Uses strict `< radius` semantics instead of inclusive `<= radius`.
    #[inline]
    pub fn exclusive_boundaries(self) -> <Self as ExclusiveBoundariesQueryBuilder>::Output
    where
        Self: ExclusiveBoundariesQueryBuilder,
    {
        <Self as ExclusiveBoundariesQueryBuilder>::exclusive_boundaries(self)
    }
}

enum WithinUnsortedBuilderIter<'a, Tree, A, T, SS, LS, D, const K: usize, const B: usize>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    D: DistanceMetricSimdBlock<A, K>,
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
pub(crate) fn boundary_accepts<O: Axis<Coord = O>>(
    boundary: BoundaryMode,
    distance: O,
    radius: O,
) -> bool {
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
    D: DistanceMetricSimdBlock<A, K>,
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
    D: DistanceMetricSimdBlock<A, K>,
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

    if sorted || max_qty.is_some() {
        results.sort_unstable();
        if let Some(max_qty) = max_qty {
            results.truncate(max_qty);
        }
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
    D: DistanceMetricSimdBlock<A, K>,
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
    pub fn query<'a>(
        &'a self,
        query: &'a [A; K],
    ) -> RootQueryBuilder<'a, Self, A, T, SS, LS, K, B> {
        QueryBuilder {
            tree: self,
            query,
            box_size: None,
            max_qty: None,
            radius: (),
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
    ) -> RootQueryBuilder<'a, Self, A, T, SS, rkyv_08::Archived<LS>, K, B> {
        QueryBuilder {
            tree: self,
            query,
            box_size: None,
            max_qty: None,
            radius: (),
            _phantom: PhantomData,
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    /// Excludes stored point coordinates from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_points_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<Exclude, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<Exclude, I, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Includes stored items in the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_items_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, Include, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, Include, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Excludes stored items from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_items_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, Exclude, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, Exclude, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Includes distances in the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_distances_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Include>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, I, Include>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }

    /// Excludes distances from the eventual query results.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn without_distances_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        Space,
        Projection<P, I, Exclude>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, Space, Projection<P, I, Exclude>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

macro_rules! impl_projection_toggle_trait {
    ($trait:ident, $method:ident, $helper:ident, $out_p:ident, $out_i:ident, $out_d:ident) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                P,
                I,
                Dp,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > $trait
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                Projection<P, I, Dp>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            D: BuilderMetric<A, K>,
            Family: ResultFamily,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Family,
                Space,
                Projection<$out_p, $out_i, $out_d>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >;

            #[inline]
            fn $method(self) -> Self::Output {
                self.$helper()
            }
        }
    };
}

impl_projection_toggle_trait!(
    WithoutPointsQueryBuilder,
    without_points,
    without_points_impl,
    Exclude,
    I,
    Dp
);
impl_projection_toggle_trait!(
    WithItemsQueryBuilder,
    with_items,
    with_items_impl,
    P,
    Include,
    Dp
);
impl_projection_toggle_trait!(
    WithoutItemsQueryBuilder,
    without_items,
    without_items_impl,
    P,
    Exclude,
    Dp
);
impl_projection_toggle_trait!(
    WithDistancesQueryBuilder,
    with_distances,
    with_distances_impl,
    P,
    I,
    Include
);
impl_projection_toggle_trait!(
    WithoutDistancesQueryBuilder,
    without_distances,
    without_distances_impl,
    P,
    I,
    Exclude
);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    /// Includes stored point coordinates in the eventual query results.
    ///
    /// This is available only for non-periodic queries.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn with_points_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<Include, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<D, Family, CartesianSpace, Projection<Include, I, Dp>, SORTED, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > WithPointsQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: BuilderMetric<A, K>,
    Family: ResultFamily,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Family,
        CartesianSpace,
        Projection<Include, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >;

    #[inline]
    fn with_points(self) -> Self::Output {
        self.with_points_impl()
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    /// Interprets subsequent queries using periodic boundary conditions.
    ///
    /// `box_size` gives the period along each axis and every value must be
    /// strictly positive.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn periodic_boundary_condition_impl(
        self,
        box_size: &'a [A; K],
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        A: PeriodicAxis,
    {
        self.rebind::<
            NoMetric,
            RootState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(Some(box_size), None, ())
    }

    /// Selects an exact nearest-neighbour query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_one_impl<D>(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestOneState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, None, D::Output::max_value())
    }

    /// Selects a k-nearest-neighbours query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_n_impl<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestNState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, Some(max_qty), D::Output::max_value())
    }

    /// Selects a radius query returning all entries within `radius`.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn within_impl<D>(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            WithinState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, None, radius)
    }

    /// Selects a radius query that keeps the best `max_qty` entries by item ordering.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn best_n_within_impl<D>(
        self,
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        BestNWithinState,
        CartesianSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        T: Content + PartialOrd + 'static,
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            BestNWithinState,
            CartesianSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(None, Some(max_qty), radius)
    }
}

macro_rules! impl_root_metric_selector_noarg {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetricSimdBlock<A, K>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self) -> Self::Output {
                self.$helper::<Dq>()
            }
        }
    };
}

macro_rules! impl_root_metric_selector_qty {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetricSimdBlock<A, K>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self, max_qty: NonZeroUsize) -> Self::Output {
                self.$helper::<Dq>(max_qty)
            }
        }
    };
}

macro_rules! impl_root_metric_selector_radius {
    ($space:ty, [$($a_bounds:tt)+], $trait:ident, $method:ident, $helper:ident, $family:ty) => {
        impl<
                'a,
                Tree,
                A: $($a_bounds)+,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > $trait<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            Dq: DistanceMetricSimdBlock<A, K>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                $family,
                $space,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn $method(self, radius: Dq::Output) -> Self::Output {
                self.$helper::<Dq>(radius)
            }
        }
    };
}

macro_rules! impl_root_best_n_within_query_builder {
    () => {
        impl<
                'a,
                Tree,
                A: Copy,
                T,
                SS,
                LS,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
                Dq,
            > BestNWithinQueryBuilder<A, K, Dq>
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                NoMetric,
                RootState,
                CartesianSpace,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            T: Content + PartialOrd + 'static,
            Dq: DistanceMetricSimdBlock<A, K>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<Dq>,
                BestNWithinState,
                CartesianSpace,
                Projection<Exclude, Include, Include>,
                true,
                false,
                K,
                B,
            >;

            #[inline]
            fn best_n_within(self, radius: Dq::Output, max_qty: NonZero<usize>) -> Self::Output {
                self.best_n_within_impl::<Dq>(radius, max_qty)
            }
        }
    };
}

impl<
        'a,
        Tree,
        A: Copy + PeriodicAxis,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > PeriodicBoundaryConditionQueryBuilder<'a, A, K>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >;

    #[inline]
    fn periodic_boundary_condition(self, box_size: &'a [A; K]) -> Self::Output {
        self.periodic_boundary_condition_impl(box_size)
    }
}
impl_root_metric_selector_noarg!(
    CartesianSpace,
    [Copy],
    NearestOneQueryBuilder,
    nearest_one,
    nearest_one_impl,
    NearestOneState
);
impl_root_metric_selector_qty!(
    CartesianSpace,
    [Copy],
    NearestNQueryBuilder,
    nearest_n,
    nearest_n_impl,
    NearestNState
);
impl_root_metric_selector_radius!(
    CartesianSpace,
    [Copy],
    WithinQueryBuilder,
    within,
    within_impl,
    WithinState
);
impl_root_best_n_within_query_builder!();

impl<
        'a,
        Tree,
        A: Copy + PeriodicAxis,
        T,
        SS,
        LS,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        NoMetric,
        RootState,
        PeriodicSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
{
    /// Selects an exact nearest-neighbour query under periodic boundary conditions.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_one_impl<D>(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestOneState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, None, D::Output::max_value())
    }

    /// Selects a periodic k-nearest-neighbours query.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn nearest_n_impl<D>(
        self,
        max_qty: NonZeroUsize,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            NearestNState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, Some(max_qty), D::Output::max_value())
    }

    /// Selects a periodic radius query returning all entries within `radius`.
    #[allow(clippy::type_complexity)]
    #[inline]
    fn within_impl<D>(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        PeriodicSpace,
        Projection<Exclude, Include, Include>,
        true,
        false,
        K,
        B,
    >
    where
        D: DistanceMetricSimdBlock<A, K>,
    {
        self.rebind::<
            SelectedMetric<D>,
            WithinState,
            PeriodicSpace,
            Projection<Exclude, Include, Include>,
            true,
            false,
        >(self.box_size, None, radius)
    }
}

impl_root_metric_selector_noarg!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    NearestOneQueryBuilder,
    nearest_one,
    nearest_one_impl,
    NearestOneState
);
impl_root_metric_selector_qty!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    NearestNQueryBuilder,
    nearest_n,
    nearest_n_impl,
    NearestNState
);
impl_root_metric_selector_radius!(
    PeriodicSpace,
    [Copy + PeriodicAxis],
    WithinQueryBuilder,
    within,
    within_impl,
    WithinState
);

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetricSimdBlock<A, K>,
{
    /// Switches exact nearest-neighbour search to approximate descent-only mode.
    #[inline]
    fn approx_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, ApproxNearestOneState, CartesianSpace, Pj, SORTED, EXCLUSIVE>(
            None,
            self.max_qty,
            self.radius,
        )
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ApproxQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetricSimdBlock<A, K>,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >;

    #[inline]
    fn approx(self) -> Self::Output {
        self.approx_impl()
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetricSimdBlock<A, K>,
{
    /// Adds a radius bound to a k-nearest-neighbours query.
    #[inline]
    fn within_impl(
        self,
        radius: D::Output,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        SORTED,
        false,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNWithinState, Space, Pj, SORTED, false>(
            self.box_size,
            self.max_qty,
            radius,
        )
    }
}

impl<
        'a,
        Tree,
        A: Copy,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > WithinQueryBuilder<A, K, D>
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    D: DistanceMetricSimdBlock<A, K>,
{
    type Output = QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        SORTED,
        false,
        K,
        B,
    >;

    #[inline]
    fn within(self, radius: D::Output) -> Self::Output {
        self.within_impl(radius)
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

macro_rules! impl_unsorted_query_builder {
    ($family:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Space,
                Pj,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > UnsortedQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                Space,
                Pj,
                true,
                EXCLUSIVE,
                K,
                B,
            >
        where
            A: Copy,
            D: DistanceMetricCore<A>,
        {
            type Output = QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                Space,
                Pj,
                false,
                EXCLUSIVE,
                K,
                B,
            >;

            #[inline]
            fn unsorted(self) -> Self::Output {
                self.unsorted_impl()
            }
        }
    };
}

impl_unsorted_query_builder!(NearestNState);

macro_rules! impl_exclusive_boundaries {
    ($family:ty, $space:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                Pj,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > ExclusiveBoundariesQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                $family,
                $space,
                Pj,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            D: BuilderMetric<A, K>,
        {
            type Output =
                QueryBuilder<'a, Tree, A, T, SS, LS, D, $family, $space, Pj, SORTED, true, K, B>;

            #[inline]
            fn exclusive_boundaries(self) -> Self::Output {
                self.rebind::<D, $family, $space, Pj, SORTED, true>(
                    self.box_size,
                    self.max_qty,
                    self.radius,
                )
            }
        }
    };
}

impl_exclusive_boundaries!(NearestNWithinState, CartesianSpace);
impl_exclusive_boundaries!(NearestNWithinState, PeriodicSpace);
impl_exclusive_boundaries!(WithinState, CartesianSpace);
impl_exclusive_boundaries!(WithinState, PeriodicSpace);
impl_exclusive_boundaries!(BestNWithinState, CartesianSpace);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, NearestNWithinState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl_unsorted_query_builder!(NearestNWithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        Space,
        Pj,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        Space,
        Pj,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Copy,
    D: DistanceMetricCore<A>,
{
    /// Returns results in traversal order instead of sorted-by-distance order.
    #[inline]
    fn unsorted_impl(
        self,
    ) -> QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        Space,
        Pj,
        false,
        EXCLUSIVE,
        K,
        B,
    > {
        self.rebind::<SelectedMetric<D>, WithinState, Space, Pj, false, EXCLUSIVE>(
            self.box_size,
            self.max_qty,
            self.radius,
        )
    }
}

impl_unsorted_query_builder!(WithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree, self.query,
            )
        } else {
            let (distance, item) = self.tree.qb_nearest_one::<D>(self.query);
            project_nearest_without_point_from_parts::<A, T, D::Output, P, I, Dp, K>(item, distance)
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        ApproxNearestOneState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K, Output = A>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<A>,
{
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, A, K>>::WANTS_POINTS {
            scan_projected_nearest_one::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree, self.query,
            )
        } else {
            let (distance, item) = self.tree.qb_approx_nearest_one::<D>(self.query);
            project_nearest_without_point_from_parts::<A, T, A, P, I, Dp, K>(item, distance)
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestOneState,
        PeriodicSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: PeriodicAxis + 'static,
    T: Content,
    SS: StemStrategy + 'static,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = QueryResultItem<P::Output, I::Output, Dp::Output>;

    #[inline]
    fn execute(self) -> QueryResultItem<P::Output, I::Output, Dp::Output> {
        let result = periodic_nearest_one_result::<Tree, A, T, SS, LS, D, K, B>(
            self.tree,
            self.query,
            self.box_size.expect("periodic builder missing box size"),
        );
        project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result)
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                None,
                self.max_qty.map(NonZeroUsize::get),
                true,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            self.tree
                .qb_nearest_n::<D>(self.query, self.max_qty.expect("max_qty missing"), true)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                None,
                self.max_qty.map(NonZeroUsize::get),
                false,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            self.tree
                .qb_nearest_n::<D>(self.query, self.max_qty.expect("max_qty missing"), false)
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                self.max_qty.map(NonZeroUsize::get),
                true,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_nearest_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    true,
                )
            } else {
                self.tree.qb_nearest_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    true,
                )
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                self.max_qty.map(NonZeroUsize::get),
                false,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_nearest_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    false,
                )
            } else {
                self.tree.qb_nearest_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                    false,
                )
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        true,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                None,
                true,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_within::<D, true>(self.query, self.radius)
            } else {
                self.tree.qb_within::<D, false>(self.query, self.radius)
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_nearest_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                Some(self.radius),
                None,
                false,
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree
                    .qb_within_unsorted::<D, true>(self.query, self.radius)
            } else {
                self.tree
                    .qb_within_unsorted::<D, false>(self.query, self.radius)
            };
            results
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >
    QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        WithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        false,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B> + 'a,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B> + 'a,
    D: DistanceMetricSimdBlock<A, K> + 'a,
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
    /// Visits each matching entry without materializing a result vector.
    #[inline]
    pub fn visit<F>(self, mut visitor: F)
    where
        F: FnMut(QueryResultItem<P::Output, I::Output, Dp::Output>),
    {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            let query_wide = self.query.map(D::widen_coord);
            for (item, point) in KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(self.tree) {
                let point_wide = point.map(D::widen_coord);
                let distance = D::dist(&point_wide, &query_wide);
                if boundary_accepts(
                    if EXCLUSIVE {
                        BoundaryMode::Exclusive
                    } else {
                        BoundaryMode::Inclusive
                    },
                    distance,
                    self.radius,
                ) {
                    visitor(Projection::<P, I, Dp>::nearest_from_parts(
                        point, item, distance,
                    ));
                }
            }
        } else if EXCLUSIVE {
            self.tree.qb_within_unsorted_visit::<D, _, true>(
                self.query,
                self.radius,
                move |result| {
                    visitor(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result))
                },
            );
        } else {
            self.tree.qb_within_unsorted_visit::<D, _, false>(
                self.query,
                self.radius,
                move |result| {
                    visitor(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>(result))
                },
            );
        }
    }

    /// Returns an iterator over matching entries without first materializing a result vector.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn iter(
        self,
    ) -> Box<dyn Iterator<Item = QueryResultItem<P::Output, I::Output, Dp::Output>> + 'a> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            let query_wide = self.query.map(D::widen_coord);
            let radius = self.radius;
            let boundary = if EXCLUSIVE {
                BoundaryMode::Exclusive
            } else {
                BoundaryMode::Inclusive
            };
            Box::new(
                KdTreeIter::<Tree, A, T, SS, LS, K, B>::new(self.tree).filter_map(
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
            let iter = if EXCLUSIVE {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Exclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, true, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            } else {
                WithinUnsortedBuilderIter::<Tree, A, T, SS, LS, D, K, B>::Inclusive(
                    WithinUnsortedIter::<_, _, _, _, _, _, false, K, B>::new(
                        self.tree,
                        self.query,
                        self.radius,
                    ),
                )
            };
            Box::new(iter.map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>))
        }
    }
}

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        NearestNState,
        PeriodicSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: PeriodicAxis + 'static,
    T: Content + Copy + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
    D::Output: Axis<Coord = D::Output>,
    P: PointProjectionField<A, K>,
    I: ProjectionField<T>,
    Dp: ProjectionField<D::Output>,
{
    type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
        periodic_nearest_results::<Tree, A, T, SS, LS, D, false, K, B>(
            self.tree,
            self.query,
            self.box_size.expect("periodic builder missing box size"),
            None,
            self.max_qty,
            true,
        )
        .into_iter()
        .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
        .collect()
    }
}

macro_rules! impl_periodic_radius_vec_execute {
    ($family:ty) => {
        impl<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                D,
                P,
                I,
                Dp,
                const SORTED: bool,
                const EXCLUSIVE: bool,
                const K: usize,
                const B: usize,
            > ExecuteQueryBuilder
            for QueryBuilder<
                'a,
                Tree,
                A,
                T,
                SS,
                LS,
                SelectedMetric<D>,
                $family,
                PeriodicSpace,
                Projection<P, I, Dp>,
                SORTED,
                EXCLUSIVE,
                K,
                B,
            >
        where
            A: PeriodicAxis + 'static,
            T: Content + Copy + PartialOrd,
            SS: StemStrategy,
            LS: LeafStrategy<A, T, SS, K, B>,
            Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeQueryOps<A, T, SS, LS, K, B>,
            D: DistanceMetricSimdBlock<A, K>,
            D::Output: Axis<Coord = D::Output>,
            P: PointProjectionField<A, K>,
            I: ProjectionField<T>,
            Dp: ProjectionField<D::Output>,
        {
            type Output = Vec<QueryResultItem<P::Output, I::Output, Dp::Output>>;

            #[inline]
            fn execute(self) -> Vec<QueryResultItem<P::Output, I::Output, Dp::Output>> {
                periodic_nearest_results::<Tree, A, T, SS, LS, D, EXCLUSIVE, K, B>(
                    self.tree,
                    self.query,
                    self.box_size.expect("periodic builder missing box size"),
                    Some(self.radius),
                    self.max_qty,
                    SORTED,
                )
                .into_iter()
                .map(project_nearest_without_point::<A, T, D::Output, P, I, Dp, K>)
                .collect()
            }
        }
    };
}

impl_periodic_radius_vec_execute!(NearestNWithinState);
impl_periodic_radius_vec_execute!(WithinState);

impl<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        D,
        P,
        I,
        Dp,
        const SORTED: bool,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    > ExecuteQueryBuilder
    for QueryBuilder<
        'a,
        Tree,
        A,
        T,
        SS,
        LS,
        SelectedMetric<D>,
        BestNWithinState,
        CartesianSpace,
        Projection<P, I, Dp>,
        SORTED,
        EXCLUSIVE,
        K,
        B,
    >
where
    A: Axis<Coord = A> + 'static,
    T: Content + Copy + Default + PartialOrd,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
    Tree: QueryBuilderTreeOps<A, T, SS, LS, K, B> + KdTreeAccessor<A, T, SS, LS, K, B>,
    D: DistanceMetricSimdBlock<A, K>,
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
    type Output = BinaryHeap<BestQueryResultItem<P::Output, I::Output, Dp::Output>>;

    #[inline]
    fn execute(self) -> BinaryHeap<BestQueryResultItem<P::Output, I::Output, Dp::Output>> {
        if <Projection<P, I, Dp> as ProjectionSpec<A, T, D::Output, K>>::WANTS_POINTS {
            scan_projected_best_results::<Tree, A, T, SS, LS, D, Projection<P, I, Dp>, K, B>(
                self.tree,
                self.query,
                self.radius,
                self.max_qty.expect("max_qty missing").get(),
                if EXCLUSIVE {
                    BoundaryMode::Exclusive
                } else {
                    BoundaryMode::Inclusive
                },
            )
        } else {
            let results = if EXCLUSIVE {
                self.tree.qb_best_n_within::<D, true>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                )
            } else {
                self.tree.qb_best_n_within::<D, false>(
                    self.query,
                    self.radius,
                    self.max_qty.expect("max_qty missing"),
                )
            };
            results
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
    fn periodic_nearest_one_wraps_and_supports_non_point_projection() {
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
    fn periodic_radius_queries_use_per_entry_semantics_and_respect_boundary_mode() {
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

    #[test]
    fn periodic_plural_queries_do_not_dedup_duplicate_items() {
        let points = [
            (7u32, [0.95, 0.5]),
            (7u32, [0.96, 0.5]),
            (9u32, [0.85, 0.5]),
        ];
        let tree = WrapTree::new_from_entries(&points).unwrap();
        let query = [0.05, 0.5];
        let box_size = [1.0, 1.0];

        let mut results: Vec<_> = tree
            .query(&query)
            .periodic_boundary_condition(&box_size)
            .within::<SquaredEuclidean<f64>>(0.02)
            .unsorted()
            .execute()
            .into_iter()
            .map(|result| result.item)
            .collect();
        results.sort_unstable();

        assert_eq!(results, vec![7, 7]);
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
