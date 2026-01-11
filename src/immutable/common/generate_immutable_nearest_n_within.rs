#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_n_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn nearest_n_within<D>(&self, query: &[A; K], dist: A, max_items: NonZero<usize>, sorted: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                let max_items = max_items.into();

                if sorted && max_items < usize::MAX {
                    if max_items <= MAX_VEC_RESULT_SIZE {
                        self.nearest_n_within_stub::<D, SortedVec<NearestNeighbour<A, T>>>(query, dist, max_items, sorted)
                    } else {
                        self.nearest_n_within_stub::<D, BinaryHeap<NearestNeighbour<A, T>>>(query, dist, max_items, sorted)
                    }
                } else {
                    self.nearest_n_within_stub::<D, Vec<NearestNeighbour<A,T>>>(query, dist, 0, sorted)
                }
            }

            fn nearest_n_within_stub<D: DistanceMetric<A, K>, H: ResultCollection<A, T>>(
                &self, query: &[A; K], dist: A, res_capacity: usize, sorted: bool
            ) -> Vec<NearestNeighbour<A, T>> {
                let mut matching_items = H::new_with_capacity(res_capacity);
                let mut off = [<A as AxisUnified>::zero(); K];

                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let stem_ordering = SO::new(stems_ptr);

                self.nearest_n_within_recurse::<D, H>(
                    query,
                    dist,
                    stem_ordering,
                    &mut matching_items,
                    &mut off,
                    <A as AxisUnified>::zero(),
                );

                if sorted {
                    matching_items.into_sorted_vec()
                } else {
                    matching_items.into_vec()
                }
            }

            #[allow(clippy::too_many_arguments)]
            fn nearest_n_within_recurse<D, R>(
                &self,
                query: &[A; K],
                radius: A,
                mut stem_ordering: SO,
                matching_items: &mut R,
                off: &mut [A; K],
                rd: A,
            ) where
                D: DistanceMetric<A, K>,
                R: ResultCollection<A, T>,
            {
                if stem_ordering.level() > i32::from(self.max_stem_level) || self.stems.is_empty() {
                    self.search_leaf_for_nearest_n_within::<D, R>(query, radius, matching_items, stem_ordering.leaf_idx());
                    return;
                }

                let dim = stem_ordering.dim();
                let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
                let is_right_child: bool = *unsafe { query.get_unchecked(dim) } >= val;

                let mut rd = rd;
                let old_off = off[dim];
                let new_off = query[dim].saturating_dist(val);
                tracing::trace!(?rd, ?new_off, ?old_off, ?off, dim);

                let farther_so = stem_ordering.branch_relative(is_right_child);

                self.nearest_n_within_recurse::<D, R>(
                    query,
                    radius,
                    stem_ordering,
                    matching_items,
                    off,
                    rd,
                );

                // Correct formula: rd_new = rd - old_off² + new_off²
                let new_sq = D::dist1(new_off, A::default());
                let old_sq = D::dist1(old_off, A::default());
                rd = rd - old_sq + new_sq;
                tracing::trace!(?off, "new rd = {}", rd);

                if rd <= radius && rd < matching_items.max_dist() {
                    tracing::trace!("ENTER: rd ({}) <= radius ({}) && rd < matching_items.max_dist() ({})", rd, radius, matching_items.max_dist());
                    off[dim] = new_off;
                    self.nearest_n_within_recurse::<D, R>(
                        query,
                        radius,
                        farther_so,
                        matching_items,
                        off,
                        rd,
                    );
                    off[dim] = old_off;
                    tracing::trace!("off[{}] = {}", dim, old_off);
                } else {
                    tracing::trace!("PRUNE: rd ({}) > radius ({}) || rd >= matching_items.max_dist() ({})", rd, radius, matching_items.max_dist());
                }
            }

            #[cfg_attr(not(feature = "no_inline"), inline)]
            fn search_leaf_for_nearest_n_within<D, R>(
                &self,
                query: &[A; K],
                radius: A,
                results: &mut R,
                leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
                R: ResultCollection<A, T>,
            {
                tracing::trace!("search_leaf_for_nearest_n_within: leaf_idx = {}", leaf_idx);
                let leaf_slice = self.get_leaf_slice(leaf_idx);

                leaf_slice.nearest_n_within::<D, R>(
                    query,
                    radius,
                    results,
                );
            }
        }
    };
}
