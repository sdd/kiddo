#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within::<D>(query, dist, usize::MAX, true)
            }
        }
    };
}
