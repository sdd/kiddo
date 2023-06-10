//! The trait that needs to be implemented by any distance metrics

/// Trait that needs to be implemented by any potential distance
/// metric to be used within queries
pub trait DistanceMetric<A, const K: usize> {
    /// returns the distance between two K-d points, as measured
    /// by a particular distance metric
    fn dist(a: &[A; K], b: &[A; K]) -> A;

    /// returns the distance between two points along a single axis,
    /// as measured by a particular distance metric.
    ///
    /// (needs to be implemented as it is used by the NN query implementations
    /// to extend the min acceptable distance for a node when recursing
    /// back up the tree)
    fn dist1(a: A, b: A) -> A;
}
