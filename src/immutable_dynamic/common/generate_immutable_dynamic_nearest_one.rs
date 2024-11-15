#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                self.nearest_one_recurse::<D>(
                    query,
                    0,
                    0,
                    NearestNeighbour {
                        distance: A::max_value(),
                        item: T::zero(),
                    },
                    &mut off,
                    A::zero(),
                    0,
                    0,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                stem_idx: usize,
                split_dim: usize,
                mut nearest: NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
                mut level: usize,
                mut leaf_idx: usize,
            ) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                // use cmov::Cmov;
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;

                if level > self.max_stem_level {
                    self.search_leaf_for_nearest::<D>(query, &mut nearest, leaf_idx as usize);
                    return nearest;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                leaf_idx <<= 1;
                let closer_leaf_idx = leaf_idx + is_right_child;
                let farther_leaf_idx = leaf_idx + (1 - is_right_child);

                let closer_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 1, level);
                let further_node_idx =  modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 0, level);

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                level += 1;
                let next_split_dim = (split_dim + 1).rem(K);

                let nearest_neighbour = self.nearest_one_recurse::<D>(
                    query,
                    closer_node_idx,
                    next_split_dim,
                    nearest,
                    off,
                    rd,
                    level,
                    closer_leaf_idx,
                );

                if nearest_neighbour < nearest {
                    nearest = nearest_neighbour;
                }

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= nearest.distance {
                    off[split_dim] = new_off;

                    let result = self.nearest_one_recurse::<D>(
                        query,
                        further_node_idx,
                        next_split_dim,
                        nearest,
                        off,
                        rd,
                        level,
                        farther_leaf_idx,
                    );
                    off[split_dim] = old_off;

                    if result < nearest {
                        nearest = result;
                    }
                }

                nearest
            }

            #[inline]
             fn search_leaf_for_nearest<D>(
                &self,
                query: &[A; K],
                nearest: &mut NearestNeighbour<A, T>,
                leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
            {
                let leaf_extent = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
                // let leaf_extent = self.leaf_extents[leaf_idx];
                let leaf_slice = $crate::float_leaf_slice::leaf_slice::LeafSlice::new(
                    array_init::array_init(|i|
                        &self.leaf_points[i][leaf_extent.start as usize..leaf_extent.end as usize]
                    ),
                    &self.leaf_items[leaf_extent.start as usize..leaf_extent.end as usize],
                );

                let mut best_item = nearest.item;
                let mut best_dist = nearest.distance;

                leaf_slice.nearest_one::<D>(
                    query,
                    &mut best_dist,
                    &mut best_item
                );

                nearest.distance = best_dist;
                nearest.item = best_item;
            }
        }
    };
}
