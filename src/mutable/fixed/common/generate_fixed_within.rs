#[doc(hidden)]
#[macro_export]
macro_rules! generate_fixed_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D, R: AxisFixed>(&self, query: &[A; K], dist: R) -> Vec<NearestNeighbour<R, T>>
            where
                D: DistanceMetricFixed<A, K, R>,
            {
                let mut matching_items = self.within_unsorted::<D, R>(query, dist);
                matching_items.sort();
                matching_items
            }
        }
    };
}
