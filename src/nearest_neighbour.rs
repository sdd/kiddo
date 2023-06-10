//! A result item returned by a query
use crate::types::Content;
use std::cmp::Ordering;

/// Represents an entry in the results of a nerest neighbour query, with `distance` being the distance of this
/// particular item from the query point, and `item` being the stored item index that was found
/// as part of the query.
#[derive(Debug, Copy, Clone)]
pub struct NearestNeighbour<A, T> {
    /// the distance of the found item from the query point according to the supplied distance metric
    pub distance: A,
    /// the stored index of an item that was found in the query
    pub item: T,
}

impl<A: PartialOrd, T: Content> Ord for NearestNeighbour<A, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A: PartialOrd, T: Content> PartialOrd for NearestNeighbour<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A: PartialEq, T: Content> Eq for NearestNeighbour<A, T> {}

impl<A: PartialEq, T: Content> PartialEq for NearestNeighbour<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}

impl<A, T: Content> From<NearestNeighbour<A, T>> for (A, T) {
    fn from(elem: NearestNeighbour<A, T>) -> Self {
        (elem.distance, elem.item)
    }
}
