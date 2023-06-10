#[macro_export]
macro_rules! generate_within_unsorted {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                self.within_unsorted_iter::<D>(query, dist).collect()
            }
        }
    };
}
