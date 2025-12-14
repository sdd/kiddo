/// Context for spatial queries, providing the query point and pruning distance.
///
/// This trait is implemented by query-specific context structs that hold
/// the query point and track the maximum distance for branch pruning during
/// backtracking search.
pub trait QueryContext<A, O, const K: usize> {
    /// Returns the query point coordinates.
    fn query(&self) -> &[A; K];

    /// Returns the current maximum distance for pruning.
    ///
    /// This is used during backtracking to prune branches that cannot contain
    /// better results than already found. For nearest neighbor queries, this
    /// returns the distance to the best point found so far.
    fn max_dist(&self) -> O;
}
