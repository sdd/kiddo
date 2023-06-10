use generator::Generator;
use crate::nearest_neighbour::NearestNeighbour;

pub struct WithinUnsortedIter<'a, A, T>(Generator<'a, (), NearestNeighbour<A, T>>);

impl<'a, A, T> WithinUnsortedIter<'a, A, T> {
    pub fn new(gen: Generator<'a, (), NearestNeighbour<A, T>>) -> Self {
        WithinUnsortedIter(gen)
    }
}

impl<'a, A, T> Iterator for WithinUnsortedIter<'a, A, T> {
    type Item = NearestNeighbour<A, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
