#[doc(hidden)]
pub mod best_neighbour;

#[doc(hidden)]
pub mod nearest_neighbour;

pub(crate) mod result_collection;

#[cfg(feature = "result_collection_stats")]
#[doc(hidden)]
pub mod result_collection_stats;
