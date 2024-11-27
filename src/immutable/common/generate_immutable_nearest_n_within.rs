#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_n_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
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
                let mut off = [A::zero(); K];

                #[cfg(not(feature = "modified_van_emde_boas"))]
                let initial_stem_idx = 1;
                #[cfg(feature = "modified_van_emde_boas")]
                let initial_stem_idx = 0;

                self.nearest_n_within_recurse::<D, H>(
                    query,
                    dist,
                    initial_stem_idx,
                    0,
                    &mut matching_items,
                    &mut off,
                    A::zero(),
                    0,
                    0,
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
                stem_idx: usize,
                split_dim: usize,
                matching_items: &mut R,
                off: &mut [A; K],
                rd: A,
                mut level: usize,
                mut leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
                R: ResultCollection<A, T>,
            {
                #[cfg(feature = "modified_van_emde_boas")]
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;

                if level > self.max_stem_level as usize || self.stems.is_empty() {
                    self.search_leaf_for_nearest_n_within::<D, R>(query, radius, matching_items, leaf_idx as usize);
                    return;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                leaf_idx <<= 1;
                let closer_leaf_idx = leaf_idx + is_right_child;
                let further_leaf_idx = leaf_idx + (1 - is_right_child);

                #[cfg(feature = "modified_van_emde_boas")]
                let closer_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 1, /*minor_*/level);
                #[cfg(feature = "modified_van_emde_boas")]
                let further_node_idx =  modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 0, /*minor_*/level);

                #[cfg(not(feature = "modified_van_emde_boas"))]
                let closer_node_idx = (stem_idx << 1) + is_right_child;
                #[cfg(not(feature = "modified_van_emde_boas"))]
                let further_node_idx = (stem_idx << 1) + 1 - is_right_child;

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                level += 1;
                let next_split_dim = (split_dim + 1).rem(K);
                // minor_level += 1;
                // minor_level.cmovnz(&0, u8::from(minor_level == 3));

                self.nearest_n_within_recurse::<D, R>(
                    query,
                    radius,
                    closer_node_idx,
                    next_split_dim,
                    matching_items,
                    off,
                    rd,
                    level,
                    closer_leaf_idx,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= radius && rd < matching_items.max_dist() {
                    off[split_dim] = new_off;
                    self.nearest_n_within_recurse::<D, R>(
                        query,
                        radius,
                        further_node_idx,
                        next_split_dim,
                        matching_items,
                        off,
                        rd,
                        level,
                        further_leaf_idx,
                    );
                    off[split_dim] = old_off;
                }
            }

            #[inline]
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
