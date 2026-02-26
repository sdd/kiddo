#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within {
    ($comments:tt) => {
            #[doc = concat!$comments]
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.within_with_condition::<D>(query, dist, true)
            }

            #[doc = concat!$comments]
            #[inline]
            pub fn within_with_condition<D>(&self, query: &[A; K], dist: A, inclusive: bool) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within_with_condition::<D>(query, dist, std::num::NonZero::new(usize::MAX).unwrap(), true, inclusive)
            }
    };
}
