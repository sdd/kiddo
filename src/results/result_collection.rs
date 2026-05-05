use std::collections::BinaryHeap;

#[cfg(any(
    feature = "buffered_result_collection",
    feature = "small_n_result_collectors"
))]
use smallvec::SmallVec;
use sorted_vec::SortedVec;

use super::nearest_neighbour::NearestNeighbour;
use crate::{Axis, BestNeighbour, Content};

#[cfg(feature = "small_n_result_collectors")]
pub(crate) const SMALL_RESULT_COLLECTION_MAX_QTY: usize = 32;

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
pub(crate) const BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY: usize = 64;

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
pub(crate) type ResultBuffer<E> = SmallVec<[E; BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY]>;

pub trait ResultCollection<O: Axis<Coord = O>, E>: Sized {
    fn with_max_qty(max_qty: usize) -> Self;
    fn max_qty(&self) -> usize;
    fn len(&self) -> usize;
    fn add(&mut self, entry: E);
    fn threshold_distance(&self) -> Option<O>;
    fn into_vec(self) -> Vec<E>;
    fn into_sorted_vec(self) -> Vec<E>;

    #[cfg(feature = "buffered_result_collection")]
    #[inline(always)]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = E>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            for entry in iter {
                self.add(entry);
            }
        }

        #[cfg(not(feature = "result_collection_stats"))]
        for entry in entries {
            self.add(entry);
        }
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len() >= self.max_qty()
    }
}

pub(crate) struct VisitorResultCollection<'a, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    visitor: &'a mut F,
    len: usize,
    _phantom: std::marker::PhantomData<(O, E)>,
}

impl<'a, O, E, F> VisitorResultCollection<'a, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    #[inline(always)]
    pub(crate) fn new(visitor: &'a mut F) -> Self {
        Self {
            visitor,
            len: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O, E, F> ResultCollection<O, E> for VisitorResultCollection<'_, O, E, F>
where
    O: Axis<Coord = O>,
    F: FnMut(E),
{
    #[inline(always)]
    fn with_max_qty(_max_qty: usize) -> Self {
        panic!("VisitorResultCollection must be constructed with VisitorResultCollection::new")
    }

    #[inline(always)]
    fn max_qty(&self) -> usize {
        usize::MAX
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn add(&mut self, entry: E) {
        self.len += 1;
        (self.visitor)(entry);
    }

    #[inline(always)]
    fn threshold_distance(&self) -> Option<O> {
        None
    }

    #[inline(always)]
    fn into_vec(self) -> Vec<E> {
        Vec::new()
    }

    #[inline(always)]
    fn into_sorted_vec(self) -> Vec<E> {
        Vec::new()
    }
}

#[doc(hidden)]
pub trait BestNeighbourResultCollection<O: Axis<Coord = O>, T: Content + PartialOrd>:
    ResultCollection<O, BestNeighbour<O, T>>
{
    fn threshold_item(&self) -> Option<T>;
}

#[derive(Debug)]
pub(crate) struct BinaryHeapResultCollection<E> {
    max_qty: usize,
    inner: BinaryHeap<E>,
}

#[derive(Debug)]
#[cfg_attr(feature = "small_n_result_collectors", allow(dead_code))]
pub(crate) struct SortedVecResultCollection<E: Ord> {
    max_qty: usize,
    inner: SortedVec<E>,
}

#[cfg(feature = "small_n_result_collectors")]
#[derive(Debug)]
pub(crate) struct SmallBinaryHeapResultCollection<E: Ord> {
    max_qty: usize,
    inner: SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
}

#[cfg(feature = "small_n_result_collectors")]
#[derive(Debug)]
pub(crate) struct SmallSortedVecResultCollection<E: Ord> {
    max_qty: usize,
    inner: SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
}

impl<E> BinaryHeapResultCollection<E> {
    #[inline(always)]
    pub(crate) fn into_inner(self) -> BinaryHeap<E> {
        self.inner
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<E: Ord> SmallBinaryHeapResultCollection<E> {
    #[inline(always)]
    pub(crate) fn into_inner(self) -> BinaryHeap<E> {
        BinaryHeap::from(self.inner.into_vec())
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn sift_up_max_heap<E: Ord>(heap: &mut [E], mut idx: usize) {
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if heap[parent] >= heap[idx] {
            break;
        }
        heap.swap(parent, idx);
        idx = parent;
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn sift_down_max_heap<E: Ord>(heap: &mut [E], mut idx: usize) {
    let len = heap.len();
    loop {
        let left = idx * 2 + 1;
        if left >= len {
            break;
        }

        let right = left + 1;
        let mut largest = left;
        if right < len && heap[right] > heap[left] {
            largest = right;
        }

        if heap[idx] >= heap[largest] {
            break;
        }

        heap.swap(idx, largest);
        idx = largest;
    }
}

#[cfg(all(
    feature = "small_n_result_collectors",
    feature = "buffered_result_collection"
))]
#[inline(always)]
fn heapify_max_heap<E: Ord>(heap: &mut [E]) {
    if heap.len() <= 1 {
        return;
    }

    let mut idx = heap.len() / 2;
    while idx > 0 {
        idx -= 1;
        sift_down_max_heap(heap, idx);
    }
}

#[cfg(feature = "small_n_result_collectors")]
#[inline(always)]
fn small_sorted_insert<E: Ord>(
    inner: &mut SmallVec<[E; SMALL_RESULT_COLLECTION_MAX_QTY]>,
    max_qty: usize,
    entry: E,
) {
    if inner.len() < max_qty {
        let insert_at = inner.partition_point(|existing| *existing <= entry);
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_sorted_insert(
            insert_at,
            inner.len() - insert_at,
        );
        inner.insert(insert_at, entry);
    } else if entry < *inner.last().unwrap() {
        let insert_at = inner.partition_point(|existing| *existing <= entry);
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_sorted_insert(
            insert_at,
            inner.len().saturating_sub(insert_at + 1),
        );
        inner.insert(insert_at, entry);
        inner.pop();
    }
}

#[cfg(feature = "buffered_result_collection")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn flush_result_buffer<O, E, R>(results: &mut R, buffer: &mut ResultBuffer<E>)
where
    O: Axis<Coord = O>,
    R: ResultCollection<O, E>,
{
    #[cfg(feature = "result_collection_stats")]
    crate::results::result_collection_stats::record_buffer_flush(buffer.len());

    if !buffer.is_empty() {
        results.add_all(buffer.drain(..));
    }
}

impl<O: Axis<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
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
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_replacement();
                *max_heap_value = entry;
            } else {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_discard();
            }
        }
    }

    #[cfg(feature = "buffered_result_collection")]
    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner = BinaryHeap::from(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner = BinaryHeap::from(combined);
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.peek().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_sorted_vec()
    }
}

impl<O: Axis<Coord = O>, T: Content + PartialOrd> ResultCollection<O, BestNeighbour<O, T>>
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
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
        } else {
            let mut max_heap_value = self.inner.peek_mut().unwrap();
            if entry < *max_heap_value {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_replacement();
                *max_heap_value = entry;
            } else {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_heap_add_discard();
            }
        }
    }

    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner = BinaryHeap::from(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner = BinaryHeap::from(combined);
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

impl<O: Axis<Coord = O>, T: Content + PartialOrd> BestNeighbourResultCollection<O, T>
    for BinaryHeapResultCollection<BestNeighbour<O, T>>
{
    #[inline(always)]
    fn threshold_item(&self) -> Option<T> {
        if self.is_full() {
            self.inner.peek().map(|n| n.item)
        } else {
            None
        }
    }
}

impl<O: Axis<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
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
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            {
                let insert_at = self.inner.partition_point(|existing| *existing <= entry);
                crate::results::result_collection_stats::record_sorted_insert(
                    insert_at,
                    self.inner.len() - insert_at,
                );
            }
            self.inner.insert(entry);
        } else if entry < *self.inner.last().unwrap() {
            #[cfg(feature = "result_collection_stats")]
            {
                let insert_at = self.inner.partition_point(|existing| *existing <= entry);
                crate::results::result_collection_stats::record_sorted_insert(
                    insert_at,
                    self.inner.len().saturating_sub(insert_at + 1),
                );
            }
            self.inner.pop();
            self.inner.push(entry);
        }
    }

    #[allow(unreachable_code)] // needed because of early return when result_collection_stats feature is enabled
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);
            self.inner = SortedVec::from_unsorted(combined);
            if self.inner.len() > self.max_qty {
                self.inner.mutate_vec(|vec| {
                    vec.truncate(self.max_qty);
                });
            }
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);
        self.inner = SortedVec::from_unsorted(combined);
        if self.inner.len() > self.max_qty {
            self.inner.mutate_vec(|vec| {
                vec.truncate(self.max_qty);
            });
        }
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
    for SmallBinaryHeapResultCollection<NearestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: NearestNeighbour<O, T>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_replacement();
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        } else {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_discard();
        }
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner.extend(combined);
            heapify_max_heap(self.inner.as_mut_slice());
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner.extend(combined);
        heapify_max_heap(self.inner.as_mut_slice());
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.first().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        let mut vec = self.inner.into_vec();
        vec.sort();
        vec
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T: Content + PartialOrd> ResultCollection<O, BestNeighbour<O, T>>
    for SmallBinaryHeapResultCollection<BestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: BestNeighbour<O, T>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();

        if self.inner.len() < self.max_qty {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_push();
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_replacement();
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        } else {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_heap_add_discard();
        }
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);

            if combined.len() > self.max_qty {
                combined.sort_unstable();
                combined.truncate(self.max_qty);
            }

            self.inner.extend(combined);
            heapify_max_heap(self.inner.as_mut_slice());
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner.extend(combined);
        heapify_max_heap(self.inner.as_mut_slice());
    }

    fn threshold_distance(&self) -> Option<O> {
        None
    }

    fn into_vec(self) -> Vec<BestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<BestNeighbour<O, T>> {
        let mut vec = self.inner.into_vec();
        vec.sort();
        vec
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T: Content + PartialOrd> BestNeighbourResultCollection<O, T>
    for SmallBinaryHeapResultCollection<BestNeighbour<O, T>>
{
    #[inline(always)]
    fn threshold_item(&self) -> Option<T> {
        if self.is_full() {
            self.inner.first().map(|n| n.item)
        } else {
            None
        }
    }
}

#[cfg(feature = "small_n_result_collectors")]
impl<O: Axis<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
    for SmallSortedVecResultCollection<NearestNeighbour<O, T>>
{
    fn with_max_qty(max_qty: usize) -> Self {
        debug_assert!(max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY);
        Self {
            max_qty,
            inner: SmallVec::new(),
        }
    }

    fn max_qty(&self) -> usize {
        self.max_qty
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn add(&mut self, entry: NearestNeighbour<O, T>) {
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_collector_add_call();
        small_sorted_insert(&mut self.inner, self.max_qty, entry);
    }

    #[allow(unreachable_code)]
    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        #[cfg(feature = "result_collection_stats")]
        {
            let iter = entries.into_iter();
            crate::results::result_collection_stats::record_collector_add_all_call(
                iter.size_hint().0,
            );
            let mut combined = std::mem::take(&mut self.inner).into_vec();
            combined.extend(iter);
            combined.sort_unstable();
            if combined.len() > self.max_qty {
                combined.truncate(self.max_qty);
            }
            self.inner.extend(combined);
            return;
        }

        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);
        combined.sort_unstable();
        if combined.len() > self.max_qty {
            combined.truncate(self.max_qty);
        }
        self.inner.extend(combined);
    }

    fn threshold_distance(&self) -> Option<O> {
        let is_full = self.is_full();
        let result = if is_full {
            self.inner.last().map(|n| n.distance)
        } else {
            None
        };
        #[cfg(feature = "result_collection_stats")]
        crate::results::result_collection_stats::record_threshold_distance_call(
            is_full,
            result.is_some(),
        );
        result
    }

    fn into_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<O, T>> {
        self.inner.into_vec()
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::*;

    const MAX_QTY: usize = 16;

    const SORTED_NEAREST_INPUTS: [NearestNeighbour<f64, u32>; MAX_QTY] = [
        NearestNeighbour {
            distance: 0.91,
            item: 91,
        },
        NearestNeighbour {
            distance: 0.12,
            item: 12,
        },
        NearestNeighbour {
            distance: 0.54,
            item: 54,
        },
        NearestNeighbour {
            distance: 0.07,
            item: 7,
        },
        NearestNeighbour {
            distance: 0.63,
            item: 63,
        },
        NearestNeighbour {
            distance: 0.33,
            item: 33,
        },
        NearestNeighbour {
            distance: 0.88,
            item: 88,
        },
        NearestNeighbour {
            distance: 0.19,
            item: 19,
        },
        NearestNeighbour {
            distance: 0.41,
            item: 41,
        },
        NearestNeighbour {
            distance: 0.02,
            item: 2,
        },
        NearestNeighbour {
            distance: 0.76,
            item: 76,
        },
        NearestNeighbour {
            distance: 0.27,
            item: 27,
        },
        NearestNeighbour {
            distance: 0.69,
            item: 69,
        },
        NearestNeighbour {
            distance: 0.58,
            item: 58,
        },
        NearestNeighbour {
            distance: 0.15,
            item: 15,
        },
        NearestNeighbour {
            distance: 0.47,
            item: 47,
        },
    ];

    const BEST_INPUTS: [BestNeighbour<f64, u32>; MAX_QTY] = [
        BestNeighbour {
            distance: 0.91,
            item: 91,
        },
        BestNeighbour {
            distance: 0.12,
            item: 12,
        },
        BestNeighbour {
            distance: 0.54,
            item: 54,
        },
        BestNeighbour {
            distance: 0.07,
            item: 7,
        },
        BestNeighbour {
            distance: 0.63,
            item: 63,
        },
        BestNeighbour {
            distance: 0.33,
            item: 33,
        },
        BestNeighbour {
            distance: 0.88,
            item: 88,
        },
        BestNeighbour {
            distance: 0.19,
            item: 19,
        },
        BestNeighbour {
            distance: 0.41,
            item: 41,
        },
        BestNeighbour {
            distance: 0.02,
            item: 2,
        },
        BestNeighbour {
            distance: 0.76,
            item: 76,
        },
        BestNeighbour {
            distance: 0.27,
            item: 27,
        },
        BestNeighbour {
            distance: 0.69,
            item: 69,
        },
        BestNeighbour {
            distance: 0.58,
            item: 58,
        },
        BestNeighbour {
            distance: 0.15,
            item: 15,
        },
        BestNeighbour {
            distance: 0.47,
            item: 47,
        },
    ];

    #[inline(always)]
    fn checksum_nearest(results: &[NearestNeighbour<f64, u32>]) -> (usize, u64, u64) {
        let mut checksum_item = 0u64;
        let mut checksum_dist = 0u64;

        for entry in results {
            checksum_item = checksum_item.wrapping_add(entry.item as u64);
            checksum_dist = checksum_dist.wrapping_add(entry.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist)
    }

    #[inline(always)]
    fn checksum_best(results: &[BestNeighbour<f64, u32>]) -> (usize, u64, u64) {
        let mut checksum_item = 0u64;
        let mut checksum_dist = 0u64;

        for entry in results {
            checksum_item = checksum_item.wrapping_add(entry.item as u64);
            checksum_dist = checksum_dist.wrapping_add(entry.distance.to_bits());
        }

        (results.len(), checksum_item, checksum_dist)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_vec_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SortedVecResultCollection::<NearestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        for entry in SORTED_NEAREST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "buffered_result_collection")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_sorted_vec_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SortedVecResultCollection::<NearestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        results.add_all(SORTED_NEAREST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_binary_heap_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            BinaryHeapResultCollection::<BestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        for entry in BEST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "buffered_result_collection")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_binary_heap_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            BinaryHeapResultCollection::<BestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        results.add_all(BEST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_sorted_vec_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallSortedVecResultCollection::<NearestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        for entry in SORTED_NEAREST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_sorted_vec_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallSortedVecResultCollection::<NearestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        results.add_all(SORTED_NEAREST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_nearest(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_binary_heap_result_collection_add_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallBinaryHeapResultCollection::<BestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        for entry in BEST_INPUTS {
            results.add(entry);
        }
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }

    #[cfg(feature = "small_n_result_collectors")]
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_small_binary_heap_result_collection_add_all_cargo_asm_hook() -> (usize, u64, u64) {
        let mut results =
            SmallBinaryHeapResultCollection::<BestNeighbour<f64, u32>>::with_max_qty(MAX_QTY);
        results.add_all(BEST_INPUTS);
        let vec = results.into_sorted_vec();
        checksum_best(&vec)
    }
}

impl<O: Axis<Coord = O>, E: Ord> ResultCollection<O, E> for Vec<E> {
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

    #[cfg(feature = "buffered_result_collection")]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = E>,
    {
        self.extend(entries);
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
