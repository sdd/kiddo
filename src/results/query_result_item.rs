//! A projected result item returned by nearest/range queries.
use std::cmp::Ordering;

/// Represents an entry in the results of a nearest or radius query.
#[derive(Debug, Copy, Clone)]
pub struct QueryResultItem<P, T, D> {
    /// The point that matched the query when point projection is enabled.
    pub point: P,
    /// The stored item that matched the query when item projection is enabled.
    pub item: T,
    /// The query distance when distance projection is enabled.
    pub distance: D,
}

impl<P, T, D: PartialOrd> Ord for QueryResultItem<P, T, D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[allow(renamed_and_removed_lints)]
#[allow(unknown_lints)]
#[allow(clippy::incorrect_partial_ord_impl_on_ord_type)]
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<P, T, D: PartialOrd> PartialOrd for QueryResultItem<P, T, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<P, T, D: PartialEq> Eq for QueryResultItem<P, T, D> {}

impl<P, T, D: PartialEq> PartialEq for QueryResultItem<P, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
