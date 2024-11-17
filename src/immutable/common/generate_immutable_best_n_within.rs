#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_best_n_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn best_n_within<D>(
                &self,
                query: &[A; K],
                dist: A,
                max_qty: usize,
            ) -> impl Iterator<Item = BestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T, K>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::new();

                self.best_n_within_recurse::<D>(
                    query,
                    dist,
                    max_qty,
                    0,
                    0,
                    &mut best_items,
                    &mut off,
                    A::zero(),
                    0,
                    0,
                );

                best_items.into_iter()
            }

            #[allow(clippy::too_many_arguments)]
            fn best_n_within_recurse<D>(
                &self,
                query: &[A; K],
                radius: A,
                max_qty: usize,
                stem_idx: usize,
                split_dim: usize,
                best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
                mut level: usize,
                mut leaf_idx: usize,
            ) where
                A: LeafSliceFloat<T, K>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;

                if level > self.max_stem_level as usize {
                    self.search_leaf_for_best_n_within::<D>(query, radius, max_qty, best_items, leaf_idx as usize);
                    return;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                leaf_idx <<= 1;
                let closer_leaf_idx = leaf_idx + is_right_child;
                let further_leaf_idx = leaf_idx + (1 - is_right_child);

                let closer_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 1, /*minor_*/level);
                let further_node_idx =  modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 0, /*minor_*/level);

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                level += 1;
                let next_split_dim = (split_dim + 1).rem(K);
                // minor_level += 1;
                // minor_level.cmovnz(&0, u8::from(minor_level == 3));

                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
                    closer_node_idx,
                    next_split_dim,
                    best_items,
                    off,
                    rd,
                    level,
                    closer_leaf_idx,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= radius {
                    off[split_dim] = new_off;
                    self.best_n_within_recurse::<D>(
                        query,
                        radius,
                        max_qty,
                        further_node_idx,
                        next_split_dim,
                        best_items,
                        off,
                        rd,
                        level,
                        further_leaf_idx,
                    );
                    off[split_dim] = old_off;
                }
            }

            #[inline]
            fn search_leaf_for_best_n_within<D>(
                &self,
                query: &[A; K],
                radius: A,
                max_qty: usize,
                results: &mut BinaryHeap<BestNeighbour<A, T>>,
                leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
            {
                let leaf_slice = self.get_leaf_slice(leaf_idx);

                leaf_slice.best_n_within::<D>(
                    query,
                    radius,
                    max_qty,
                    results,
                );
            }
        }
    };
}
