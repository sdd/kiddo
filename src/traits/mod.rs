//! Traits used by [`KdTree`](crate::kd_tree::KdTree).

/// Trait for coordinate/axis types
#[doc(hidden)]
pub mod axis;
#[doc(inline)]
pub use axis::Axis;

/// Trait required for items stored in the tree
#[doc(hidden)]
pub mod content;
#[doc(inline)]
pub use content::Content;

/// Advanced traits for extending or specializing distance metrics.
pub mod dist;

/// Advanced traits for kd-tree query access and stem-to-leaf resolution.
pub mod kd_tree;

/// Trait implemented by leaf strategies (determines how leaf storage is laid out)
pub mod leaf_strategy;
#[doc(hidden)]
pub use leaf_strategy::{
    ConstructibleLeafStrategy, Immutable, LeafStrategy, Mutability, Mutable, MutableLeafStrategy,
};

///Trait implemented by stem strategy (determines tree stem ordering and traversal)
#[doc(hidden)]
pub mod stem_strategy;
#[doc(inline)]
pub use stem_strategy::StemStrategy;
