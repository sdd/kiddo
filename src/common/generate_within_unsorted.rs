#[doc(hidden)]
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
                let mut off = [A::zero(); K];
                let mut matching_items = Vec::new();
                let root_index: IDX = *transform(&self.root_index);

                unsafe {
                    self.within_unsorted_recurse::<D>(
                        query,
                        dist,
                        root_index,
                        0,
                        &mut matching_items,
                        &mut off,
                        A::zero(),
                    );
                }

                matching_items
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn within_unsorted_recurse<D>(
                &self,
                query: &[A; K],
                radius: A,
                curr_node_idx: IDX,
                split_dim: usize,
                matching_items: &mut Vec<NearestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
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

                            if distance < radius {
                                let item = unsafe { leaf_node.content_items.get_unchecked(idx) };
                                let item = *transform(item);

                                matching_items.push(NearestNeighbour {
                                    distance,
                                    item,
                                })
                            }
                        });
                }
            }
        }
    };
}
