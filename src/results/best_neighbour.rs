//! Back-compat alias for the default best-within query result shape.
pub use crate::results::best_query_result_item::BestQueryResultItem;

/// Default best-within query result shape.
pub type BestNeighbour<A, T> = BestQueryResultItem<(), T, A>;
