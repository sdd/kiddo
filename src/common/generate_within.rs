#[macro_export]
macro_rules! generate_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut matching_items: BinaryHeap<NearestNeighbour<A, T>> = self.within_unsorted_iter::<D>(query, dist).collect();

                matching_items.into_sorted_vec()
            }
        }
    };
}
