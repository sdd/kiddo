#[macro_export]
macro_rules! generate_immutable_approx_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
            where
                D: DistanceMetric<A, K>,
            {
                let mut split_dim = 0;
                let mut stem_idx = 1;
                let mut best_item = T::zero();
                let mut best_dist = A::max_value();

                let stem_len = self.stems.len();

                while stem_idx < stem_len {
                    let left_child_idx = stem_idx << 1;
                    self.prefetch_stems(left_child_idx);

                    let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                    let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim) } >= val);

                    stem_idx = left_child_idx + is_right_child;

                    split_dim += 1;
                    split_dim %= K;
                }

                let leaf_node = unsafe { self.leaves.get_unchecked(stem_idx - stem_len) };
                // let leaf_node = &self.leaves[leaf_idx];

                leaf_node
                    .content_points
                    .iter()
                    .enumerate()
                    .take(leaf_node.size as usize)
                    .for_each(|(idx, entry)| {
                        let dist = D::dist(query, entry);
                        if dist < best_dist {
                            best_dist = dist;
                            best_item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                            // *best_item = leaf_node.content_items[idx]
                        }
                    });

                NearestNeighbour {
                    distance: best_dist,
                    item: best_item,
                }
            }
        }
    };
}
