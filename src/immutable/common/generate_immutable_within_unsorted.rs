#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within_unsorted {
    ($comments:tt) => {
            #[doc = concat!$comments]
            #[inline]
            pub fn within_unsorted<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.within_unsorted_exclusive::<D>(query, dist, true)
            }

            #[doc = concat!$comments]
            #[inline]
            pub fn within_unsorted_exclusive<D>(&self, query: &[A; K], dist: A, inclusive: bool) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within_exclusive::<D>(query, dist, std::num::NonZero::new(usize::MAX).unwrap(), false, inclusive)
            }
    };
}
