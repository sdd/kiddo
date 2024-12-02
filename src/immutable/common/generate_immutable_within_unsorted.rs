#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within_unsorted {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within::<D>(query, dist, std::num::NonZero::new(usize::MAX).unwrap(), false)
            }
        }
    };
}
