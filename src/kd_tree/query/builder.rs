#![allow(private_bounds)]

use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use std::num::NonZeroUsize;

use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
#[cfg(feature = "rkyv_08")]
use crate::kd_tree::ArchivedKdTree;
use crate::kd_tree::{KdTree, KdTreeAccessor, WithinUnsortedIter};
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, BestNeighbour, Content, LeafStrategy, NearestNeighbour, StemStrategy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryMode {
    Inclusive,
    Exclusive,
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
        F: FnMut(NearestNeighbour<D::Output, T>);

    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        self.within_unsorted_visit_impl::<D, F, EXCLUSIVE>(query, radius, visitor)
    }

    #[inline]
    fn qb_best_n_within<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        radius: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<D::Output, T>>
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
    type Item = NearestNeighbour<D::Output, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inclusive(iter) => iter.next(),
            Self::Exclusive(iter) => iter.next(),
        }
    }
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
    pub fn execute(self) -> (D::Output, T) {
        self.tree.qb_nearest_one::<D>(self.query)
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
    /// Executes the approximate nearest-neighbour query.
    #[inline]
    pub fn execute(self) -> (D::Output, T) {
        self.tree.qb_approx_nearest_one::<D>(self.query)
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
        self.tree.qb_nearest_n::<D>(self.query, self.max_qty, true)
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
        self.tree.qb_nearest_n::<D>(self.query, self.max_qty, false)
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
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
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    NearestNWithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
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
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    BestNWithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> BinaryHeap<BestNeighbour<D::Output, T>> {
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
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    WithinQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
        match self.boundary {
            BoundaryMode::Inclusive => self.tree.qb_within::<D, false>(self.query, self.radius),
            BoundaryMode::Exclusive => self.tree.qb_within::<D, true>(self.query, self.radius),
        }
    }
}

impl<'a, Tree, A: Copy, T, SS, LS, D, const K: usize, const B: usize>
    WithinUnsortedQuery<'a, Tree, A, T, SS, LS, D, K, B>
where
    D: KdTreeDistanceMetric<A, K>,
{
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
    pub fn execute(self) -> Vec<NearestNeighbour<D::Output, T>> {
        match self.boundary {
            BoundaryMode::Inclusive => self
                .tree
                .qb_within_unsorted::<D, false>(self.query, self.radius),
            BoundaryMode::Exclusive => self
                .tree
                .qb_within_unsorted::<D, true>(self.query, self.radius),
        }
    }

    /// Streams unsorted within-radius results directly to a visitor.
    #[inline]
    pub fn visit<F>(self, visitor: F)
    where
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        match self.boundary {
            BoundaryMode::Inclusive => {
                self.tree
                    .qb_within_unsorted_visit::<D, F, false>(self.query, self.radius, visitor)
            }
            BoundaryMode::Exclusive => {
                self.tree
                    .qb_within_unsorted_visit::<D, F, true>(self.query, self.radius, visitor)
            }
        }
    }

    /// Returns an iterator over unsorted within-radius results.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = NearestNeighbour<D::Output, T>> + 'a {
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
    }
}
