//! Back-compat alias for the default nearest/range query result shape.
pub use crate::results::query_result_item::QueryResultItem;

/// Default nearest/range query result shape.
pub type NearestNeighbour<A, T> = QueryResultItem<(), T, A>;
