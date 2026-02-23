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

    // TOOO: investigate into whether prune_on_equal_max_dist can be removed
    /// Returns true if branches with `rd == max_dist` should be pruned.
    ///
    /// Nearest-one queries can safely prune equality and gain performance.
    /// Radius-based queries generally need to keep equality (boundary points).
    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        false
    }
}
