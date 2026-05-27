//! A projected result item returned by best-within queries.
use std::cmp::Ordering;

use crate::Content;

/// Represents an entry in the results of a best-within query.
#[derive(Debug, Copy, Clone)]
pub struct BestQueryResultItem<P, T, D> {
    /// The point that matched the query when point projection is enabled.
    pub point: P,
    /// The stored item that matched the query.
    pub item: T,
    /// The query distance when distance projection is enabled.
    pub distance: D,
}

impl<P, T: Content + PartialOrd, D: PartialOrd> Ord for BestQueryResultItem<P, T, D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[allow(renamed_and_removed_lints)]
#[allow(unknown_lints)]
#[allow(clippy::incorrect_partial_ord_impl_on_ord_type)]
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<P, T: Content + PartialOrd, D: PartialOrd> PartialOrd for BestQueryResultItem<P, T, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.item.partial_cmp(&other.item)
    }
}

impl<P, T: Content + PartialOrd, D: PartialEq> Eq for BestQueryResultItem<P, T, D> {}

impl<P, T: Content + PartialOrd, D: PartialEq> PartialEq for BestQueryResultItem<P, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}
