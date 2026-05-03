//! Traits used by [`KdTree`](crate::KdTree).

/// Trait for coordinate/axis types
pub mod axis;

/// Trait required for items stored in the tree
pub mod content;

/// Trait for distance metrics
pub mod distance_metric;

/// Trait implemented by leaf strategies (determines how leaf storage is laid out)
pub mod leaf_strategy;

mod query_context;
///Trait implemented by stem strategie (determines tree stem ordering and traversal)
pub mod stem_strategy;

// pub mod traits_unified_2;
