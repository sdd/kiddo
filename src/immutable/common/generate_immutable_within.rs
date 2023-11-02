#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: BestFromDists<T, B>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within::<D>(query, dist, usize::MAX, true)
            }
        }
    };
}

/* #[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                let mut matching_items = self.within_unsorted::<D>(query, dist);
                matching_items.sort();
                matching_items
            }
        }
    };
} */
