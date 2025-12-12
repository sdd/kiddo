use crate::nearest_neighbour::NearestNeighbour;
use crate::traits_unified_2::AxisUnified;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

pub trait ResultCollection<A: AxisUnified<Coord = A>, T> {
    fn new_with_capacity(capacity: usize) -> Self;
    fn add(&mut self, entry: NearestNeighbour<A, T>);
    fn max_dist(&self) -> A;
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>>;
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>>;
}

impl<A: AxisUnified<Coord = A>, T> ResultCollection<A, T> for BinaryHeap<NearestNeighbour<A, T>> {
    fn new_with_capacity(capacity: usize) -> Self {
        BinaryHeap::with_capacity(capacity)
    }
    fn add(&mut self, entry: NearestNeighbour<A, T>) {
        let k = self.capacity();
        if self.len() < k {
            self.push(entry);
        } else {
            let mut max_heap_value = self.peek_mut().unwrap();
            if entry < *max_heap_value {
                *max_heap_value = entry;
            }
        }
    }
    fn max_dist(&self) -> A {
        if self.len() < self.capacity() {
            A::max_value()
        } else {
            self.peek().map_or(A::max_value(), |n| n.distance)
        }
    }
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        BinaryHeap::into_vec(self)
    }
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
        BinaryHeap::into_sorted_vec(self)
    }
}

impl<A: AxisUnified<Coord = A>, T> ResultCollection<A, T> for Vec<NearestNeighbour<A, T>> {
    fn new_with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn add(&mut self, entry: NearestNeighbour<A, T>) {
        self.push(entry)
    }

    fn max_dist(&self) -> A {
        A::max_value()
    }

    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self
    }

    fn into_sorted_vec(mut self) -> Vec<NearestNeighbour<A, T>> {
        self.sort();
        self
    }
}

impl<A: AxisUnified<Coord = A>, T> ResultCollection<A, T> for SortedVec<NearestNeighbour<A, T>> {
    fn new_with_capacity(capacity: usize) -> Self {
        SortedVec::with_capacity(capacity)
    }

    fn add(&mut self, entry: NearestNeighbour<A, T>) {
        let len = self.len();
        if len < self.capacity() {
            self.insert(entry);
        } else if entry < *self.last().unwrap() {
            self.pop();
            self.push(entry);
        }
    }

    fn max_dist(&self) -> A {
        if self.len() < self.capacity() {
            A::max_value()
        } else {
            self.last().map_or(A::max_value(), |n| n.distance)
        }
    }

    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.into_vec()
    }

    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.into_vec()
    }
}
