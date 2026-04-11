use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits_unified_2::AxisUnified;
use crate::BestNeighbour;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

pub trait ResultCollection<O: AxisUnified<Coord = O>, E>: Sized {
    fn with_max_qty(max_qty: usize) -> Self;
    fn max_qty(&self) -> usize;
    fn len(&self) -> usize;
    fn add(&mut self, entry: E);
    fn threshold_distance(&self) -> Option<O>;
    fn into_vec(self) -> Vec<E>;
    fn into_sorted_vec(self) -> Vec<E>;

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len() >= self.max_qty()
    }
}

#[derive(Debug)]
pub(crate) struct BinaryHeapResultCollection<E> {
    max_qty: usize,
    inner: BinaryHeap<E>,
}

#[derive(Debug)]
pub(crate) struct SortedVecResultCollection<E: Ord> {
    max_qty: usize,
    inner: SortedVec<E>,
}

impl<E> BinaryHeapResultCollection<E> {
    #[inline(always)]
    pub(crate) fn into_inner(self) -> BinaryHeap<E> {
        self.inner
    }
}

impl<O: AxisUnified<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
    for BinaryHeapResultCollection<NearestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: BinaryHeap::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: NearestNeighbour<O, T>) {
        if self.inner.len() < self.max_qty {
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                *max_heap_value = entry;
            }
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        if self.is_full() {
            self.inner.peek().map(|n| n.distance)
        } else {
            None
        }
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_sorted_vec()
    }
}

impl<O: AxisUnified<Coord = O>, T: Content> ResultCollection<O, BestNeighbour<O, T>>
    for BinaryHeapResultCollection<BestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: BinaryHeap::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: BestNeighbour<O, T>) {
        if self.inner.len() < self.max_qty {
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                *max_heap_value = entry;
            }
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<BestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<BestNeighbour<O, T>> {
        self.inner.into_sorted_vec()
    }
}

impl<O: AxisUnified<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
    for SortedVecResultCollection<NearestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        Self {
            max_qty,
            inner: SortedVec::with_capacity(max_qty),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: NearestNeighbour<O, T>) {
        if self.inner.len() < self.max_qty {
            self.inner.insert(entry);
        } else if entry < *self.inner.last().unwrap() {
            self.inner.pop();
            self.inner.push(entry);
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        if self.is_full() {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        }
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }
}

impl<O: AxisUnified<Coord = O>, E: Ord> ResultCollection<O, E> for Vec<E> {
    fn with_max_qty(max_qty: usize) -> Self {
        if max_qty == usize::MAX {
            Vec::new()
        } else {
            Vec::with_capacity(max_qty)
        }
    }

    fn max_qty(&self) -> usize {
        usize::MAX
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn add(&mut self, entry: E) {
        self.push(entry)
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<E> {
        self
    }

    fn into_sorted_vec(mut self) -> Vec<E> {
        self.sort();
        self
    }
}
