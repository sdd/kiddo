//! A result item returned by a query
use crate::float::kdtree::Axis;
use std::cmp::Ordering;

/// Represents an entry in the results of a query, with `distance` being the distance of this
/// particular item from the query point, and `item` being the stored item index that was found
/// as part of the query.
#[derive(Debug)]
pub struct Neighbour<A, T> {
    /// the distance of the found item from the query point according to the supplied distance metric
    pub distance: A,
    /// the stored index of an item that was found in the query
    pub item: T,
}

impl<A: Axis, T> Ord for Neighbour<A, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A: Axis, T> PartialOrd for Neighbour<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A: Axis, T> Eq for Neighbour<A, T> {}

impl<A: Axis, T> PartialEq for Neighbour<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A: Axis, T> PartialEq<A> for Neighbour<A, T> {
    fn eq(&self, other: &A) -> bool {
        self.distance == *other
    }
}

impl<A: Axis, T> From<Neighbour<A, T>> for (A, T) {
    fn from(elem: Neighbour<A, T>) -> Self {
        (elem.distance, elem.item)
    }
}
