#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_n_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_n_within<D>(&self, query: &[A; K], dist: A, max_items: usize, sorted: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
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

                self.nearest_n_within_recurse::<D, H>(
                    query,
                    dist,
                    1,
                    0,
                    &mut matching_items,
                    &mut off,
                    A::zero(),
                );

                if sorted {
                    matching_items.into_sorted_vec()
                } else {
                    matching_items.into_vec()
                }
            }

            #[allow(clippy::too_many_arguments)]
            fn nearest_n_within_recurse<D, R: ResultCollection<A, T>>(
                &self,
                query: &[A; K],
                radius: A,
                stem_idx: usize,
                split_dim: usize,
                matching_items: &mut R,
                off: &mut [A; K],
                rd: A,
            ) where
                D: DistanceMetric<A, K>,
            {
                if stem_idx >= self.stems.len() {
                    let leaf_node = &self.leaves[stem_idx - self.stems.len()];

                    let mut acc = [A::zero(); B];
                    (0..K).step_by(1).for_each(|dim| {
                        let qd = [query[dim]; B];

                        (0..B).step_by(1).for_each(|idx| {
                            acc[idx] += (leaf_node.content_points[dim][idx] - qd[idx])
                                    * (leaf_node.content_points[dim][idx] - qd[idx]);
                        });
                    });

                    acc
                        .iter()
                        .enumerate()
                        .take(leaf_node.size as usize)
                        .for_each(|(idx, &distance)| {

                            if distance < radius {
                                matching_items.add(NearestNeighbour {
                                    distance,
                                    item: *unsafe { leaf_node.content_items.get_unchecked(idx) },
                                });
                            }
                        });

                    return;
                }

                let left_child_idx = stem_idx << 1;
                self.prefetch_stems(left_child_idx);

                let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                // let val = self.stems[stem_idx];

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
                // let is_left_child = usize::from(query[split_dim] < val);

                let closer_node_idx = left_child_idx + (1 - is_left_child);
                let further_node_idx = left_child_idx + is_left_child;

                let next_split_dim = (split_dim + 1).rem(K);

                self.nearest_n_within_recurse::<D, R>(
                    query,
                    radius,
                    closer_node_idx,
                    next_split_dim,
                    matching_items,
                    off,
                    rd,
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
                    );
                    off[split_dim] = old_off;
                }
            }
        }
    };
}
