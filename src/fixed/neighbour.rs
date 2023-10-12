//! A result item returned by a query
use crate::fixed::kdtree::Axis;
use crate::types::Content;
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

impl<A: Axis, T: Content> Ord for Neighbour<A, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A: Axis, T: Content> PartialOrd for Neighbour<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<A: Axis, T: Content> Eq for Neighbour<A, T> {}

impl<A: Axis, T: Content> PartialEq for Neighbour<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<A: Axis, T: Content> PartialEq<A> for Neighbour<A, T> {
    fn eq(&self, other: &A) -> bool {
        self.distance == *other
    }
}

impl<A: Axis, T: Content> From<Neighbour<A, T>> for (A, T) {
    fn from(elem: Neighbour<A, T>) -> Self {
        (elem.distance, elem.item)
    }
}
