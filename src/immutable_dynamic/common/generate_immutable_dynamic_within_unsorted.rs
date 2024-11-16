#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_within_unsorted {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,            {
                self.nearest_n_within::<D>(query, dist, usize::MAX, false)
            }
        }
    };
}

/* #[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_within_unsorted {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut matching_items = Vec::new();

                self.within_unsorted_recurse::<D>(
                    query,
                    dist,
                    1,
                    0,
                    &mut matching_items,
                    &mut off,
                    A::zero(),
                );

                matching_items
            }

            #[allow(clippy::too_many_arguments)]
            fn within_unsorted_recurse<D>(
                &self,
                query: &[A; K],
                radius: A,
                stem_idx: usize,
                split_dim: usize,
                matching_items: &mut Vec<NearestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) where
                D: DistanceMetric<A, K>,
            {
                if stem_idx >= self.stems.len() {
                    let leaf_node = &self.leaves[stem_idx - self.stems.len()];

                    leaf_node
                        .content_points
                        .iter()
                        .enumerate()
                        .take(leaf_node.size as usize)
                        .for_each(|(idx, entry)| {
                            let distance = D::dist(query, entry);

                            if distance < radius {
                                matching_items.push(NearestNeighbour {
                                    distance,
                                    item: *unsafe { leaf_node.content_items.get_unchecked(idx) },
                                });
                            }
                        });

                    return;
                }

                let left_child_idx = stem_idx << 1;

                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
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

                self.within_unsorted_recurse::<D>(
                    query,
                    radius,
                    closer_node_idx,
                    next_split_dim,
                    matching_items,
                    off,
                    rd,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= radius {
                    off[split_dim] = new_off;
                    self.within_unsorted_recurse::<D>(
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
 */
