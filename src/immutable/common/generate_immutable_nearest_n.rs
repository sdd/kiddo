#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_n {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_n<D>(&self, query: &[A; K], qty: usize) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,
            {
                self.nearest_n_within::<D>(query, A::infinity(), qty, true)
            }
        }
    };
}
