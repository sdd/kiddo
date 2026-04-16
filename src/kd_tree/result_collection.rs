use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits_unified_2::AxisUnified;
use crate::BestNeighbour;
#[cfg(any(
    feature = "buffered_result_collection",
    feature = "small_n_result_collectors"
))]
use smallvec::SmallVec;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

#[cfg(feature = "small_n_result_collectors")]
pub(crate) const SMALL_RESULT_COLLECTION_MAX_QTY: usize = 32;

#[cfg(feature = "buffered_result_collection")]
pub(crate) const BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY: usize = 64;

#[cfg(feature = "buffered_result_collection")]
pub(crate) type ResultBuffer<E> = SmallVec<[E; BUFFERED_RESULT_COLLECTION_INLINE_CAPACITY]>;

pub trait ResultCollection<O: AxisUnified<Coord = O>, E>: Sized {
    fn with_max_qty(max_qty: usize) -> Self;
    fn max_qty(&self) -> usize;
    fn len(&self) -> usize;
    fn add(&mut self, entry: E);
    fn threshold_distance(&self) -> Option<O>;
    fn into_vec(self) -> Vec<E>;
    fn into_sorted_vec(self) -> Vec<E>;

    #[inline(always)]
    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = E>,
    {
        for entry in entries {
            self.add(entry);
        }
    }

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

#[cfg(feature = "small_n_result_collectors")]
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
        inner.insert(insert_at, entry);
    } else if entry < *inner.last().unwrap() {
        let insert_at = inner.partition_point(|existing| *existing <= entry);
        inner.insert(insert_at, entry);
        inner.pop();
    }
}

#[cfg(feature = "buffered_result_collection")]
#[inline(always)]
pub(crate) fn flush_result_buffer<O, E, R>(results: &mut R, buffer: &mut ResultBuffer<E>)
where
    O: AxisUnified<Coord = O>,
    R: ResultCollection<O, E>,
{
    if !buffer.is_empty() {
        results.add_all(buffer.drain(..));
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

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);

        if combined.len() > self.max_qty {
            combined.sort_unstable();
            combined.truncate(self.max_qty);
        }

        self.inner = BinaryHeap::from(combined);
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

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestNeighbour<O, T>>,
    {
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

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
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

#[cfg(feature = "small_n_result_collectors")]
impl<O: AxisUnified<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
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
        if self.inner.len() < self.max_qty {
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        }
    }

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
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
        if self.is_full() {
            self.inner.first().map(|n| n.distance)
        } else {
            None
        }
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
impl<O: AxisUnified<Coord = O>, T: Content> ResultCollection<O, BestNeighbour<O, T>>
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
        if self.inner.len() < self.max_qty {
            self.inner.push(entry);
            let len = self.inner.len();
            sift_up_max_heap(self.inner.as_mut_slice(), len - 1);
        } else if entry < self.inner[0] {
            self.inner[0] = entry;
            sift_down_max_heap(self.inner.as_mut_slice(), 0);
        }
    }

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = BestNeighbour<O, T>>,
    {
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
impl<O: AxisUnified<Coord = O>, T> ResultCollection<O, NearestNeighbour<O, T>>
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
        small_sorted_insert(&mut self.inner, self.max_qty, entry);
    }

    fn add_all<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = NearestNeighbour<O, T>>,
    {
        let mut combined = std::mem::take(&mut self.inner).into_vec();
        combined.extend(entries);
        combined.sort_unstable();
        if combined.len() > self.max_qty {
            combined.truncate(self.max_qty);
        }
        self.inner.extend(combined);
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
