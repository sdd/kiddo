#[doc(hidden)]
#[macro_export]
macro_rules! generate_nearest_n_within_unsorted {
    ($comments:tt) => {
            #[doc = concat!$comments]
            #[inline]
            pub fn nearest_n_within<D>(&self, query: &[A; K], dist: A, max_items: std::num::NonZero<usize>, sorted: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                self.nearest_n_within_exclusive::<D>(query, dist, max_items, sorted, true)
            }

            #[doc = concat!$comments]
            #[inline]
            pub fn nearest_n_within_exclusive<D>(&self, query: &[A; K], dist: A, max_items: std::num::NonZero<usize>, sorted: bool, inclusive: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                // Like [`nearest_n_within`] but allows controlling boundary inclusiveness.
                //
                // When `inclusive` is true, points at exactly the maximum distance are included.
                // When false, only points strictly less than the maximum distance are included.
                if sorted || max_items < std::num::NonZero::new(usize::MAX).unwrap() {
                    if max_items <= std::num::NonZero::new(MAX_VEC_RESULT_SIZE).unwrap() {
                        self.nearest_n_within_stub::<D, SortedVec<NearestNeighbour<A, T>>>(query, dist, max_items.get(), sorted, inclusive)
                    } else {
                        self.nearest_n_within_stub::<D, BinaryHeap<NearestNeighbour<A, T>>>(query, dist, max_items.get(), sorted, inclusive)
                    }
                } else {
                    self.nearest_n_within_stub::<D, Vec<NearestNeighbour<A,T>>>(query, dist, 0, sorted, inclusive)
                }
            }

            fn nearest_n_within_stub<D: DistanceMetric<A, K>, H: ResultCollection<A, T>>(
                &self, query: &[A; K], dist: A, res_capacity: usize, sorted: bool, inclusive: bool
            ) -> Vec<NearestNeighbour<A, T>> {
                let mut matching_items = H::new_with_capacity(res_capacity);
                let mut off = [A::zero(); K];
                let root_index: IDX = *transform(&self.root_index);

                unsafe {
                    self.nearest_n_within_unsorted_recurse::<D, H>(
                        query,
                        dist,
                        root_index,
                        0,
                        &mut matching_items,
                        &mut off,
                        A::zero(),
                        inclusive,
                    );
                }

                if sorted {
                    matching_items.into_sorted_vec()
                } else {
                    matching_items.into_vec()
                }
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn nearest_n_within_unsorted_recurse<D, R: ResultCollection<A, T>>(
                &self,
                query: &[A; K],
                radius: A,
                curr_node_idx: IDX,
                split_dim: usize,
                matching_items: &mut R,
                off: &mut [A; K],
                rd: A,
                inclusive: bool,
            ) where
                D: DistanceMetric<A, K>,
            {
                if is_stem_index(curr_node_idx) {
                    let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());
                    let split_val: A = *transform(&node.split_val);
                    let node_left: IDX = *transform(&node.left);
                    let node_right: IDX = *transform(&node.right);

                    let mut rd = rd;
                    let old_off = off[split_dim];
                    let new_off = query[split_dim].saturating_dist(split_val);

                    let [closer_node_idx, further_node_idx] =
                        if *query.get_unchecked(split_dim) < split_val {
                            [node_left, node_right]
                        } else {
                            [node_right, node_left]
                        };
                    let next_split_dim = (split_dim + 1).rem(K);

                    self.nearest_n_within_unsorted_recurse::<D, R>(
                        query,
                        radius,
                        closer_node_idx,
                        next_split_dim,
                        matching_items,
                        off,
                        rd,
                        inclusive,
                    );

                    rd = D::accumulate(rd, D::dist1(new_off, old_off));

                    if if inclusive { rd <= radius } else { rd < radius } {
                        off[split_dim] = new_off;
                        self.nearest_n_within_unsorted_recurse::<D, R>(
                            query,
                            radius,
                            further_node_idx,
                            next_split_dim,
                            matching_items,
                            off,
                            rd,
                            inclusive,
                        );
                        off[split_dim] = old_off;
                    }
                } else {
                    let leaf_node = self
                        .leaves
                        .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

                    let size: IDX = *transform(&leaf_node.size);

                    leaf_node
                        .content_points
                        .iter()
                        .enumerate()
                        .take(size.az::<usize>())
                        .for_each(|(idx, entry)| {
                            let distance = D::dist(query, transform(entry));

                            if if inclusive { distance <= radius } else { distance < radius } {
                                let item = unsafe { leaf_node.content_items.get_unchecked(idx) };
                                let item = *transform(item);

                                matching_items.add(NearestNeighbour {
                                    distance,
                                    item,
                                })
                            }
                        });
                }
            }
    };
}
