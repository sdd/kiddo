#[doc(hidden)]
#[macro_export]
macro_rules! generate_within {
    ($comments:tt) => {
            #[doc = concat!$comments]
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                self.within_exclusive::<D>(query, dist, true)
            }

            #[doc = concat!$comments]
            #[inline]
            pub fn within_exclusive<D>(&self, query: &[A; K], dist: A, inclusive: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                let mut matching_items = self.within_unsorted_exclusive::<D>(query, dist, inclusive);
                matching_items.sort();
                matching_items
            }
    };
}
