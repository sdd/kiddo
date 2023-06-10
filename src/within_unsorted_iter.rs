//! Iterator object returned by within_unsorted_iter
use crate::nearest_neighbour::NearestNeighbour;
use generator::Generator;

/// Iterator object returned by within_unsorted_iter
pub struct WithinUnsortedIter<'a, A, T>(Generator<'a, (), NearestNeighbour<A, T>>);

impl<'a, A, T> WithinUnsortedIter<'a, A, T> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbour<A, T>>) -> Self {
        WithinUnsortedIter(gen)
    }
}

impl<'a, A, T> Iterator for WithinUnsortedIter<'a, A, T> {
    type Item = NearestNeighbour<A, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
