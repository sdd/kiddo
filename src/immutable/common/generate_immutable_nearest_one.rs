#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut result = NearestNeighbour {
                    distance: A::max_value(),
                    item: T::zero(),
                };

                #[cfg(not(feature = "modified_van_emde_boas"))]
                let initial_stem_idx = 1;
                #[cfg(feature = "modified_van_emde_boas")]
                let initial_stem_idx = 0;

                #[cfg(not(feature = "modified_van_emde_boas"))]
                self.nearest_one_recurse::<D>(
                    query,
                    initial_stem_idx,
                    0,
                    &mut result,
                    &mut off,
                    A::zero(),
                    0,
                    0,
                );

                #[cfg(feature = "modified_van_emde_boas")]
                self.nearest_one_recurse::<D>(
                    query,
                    initial_stem_idx,
                    0,
                    &mut result,
                    &mut off,
                    A::zero(),
                    0,
                    0,
                    0,
                );

                result
            }

            #[allow(clippy::too_many_arguments)]
            #[cfg(feature = "modified_van_emde_boas")]
            fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                stem_idx: u32,
                split_dim: usize,
                nearest: &mut NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
                mut level: i32,
                mut minor_level: u32,
                mut leaf_idx: u32,
            )
                where
                    D: DistanceMetric<A, K>,
            {
                use cmov::Cmov;
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;

                if level > self.max_stem_level || self.stems.is_empty() {
                    self.search_leaf_for_nearest_one::<D>(query, nearest, leaf_idx as usize);
                    return;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                let is_right_child = u32::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                leaf_idx <<= 1;
                let closer_leaf_idx = leaf_idx + is_right_child;
                let farther_leaf_idx = leaf_idx + (1 - is_right_child);

                let closer_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 1, minor_level);
                let further_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx, is_right_child == 0, minor_level);

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                level += 1;
                minor_level += 1;
                minor_level.cmovnz(&0, u8::from(minor_level == 3));

                let next_split_dim = (split_dim + 1).rem(K);

                self.nearest_one_recurse::<D>(
                    query,
                    closer_node_idx,
                    next_split_dim,
                    nearest,
                    off,
                    rd,
                    level,
                    minor_level,
                    closer_leaf_idx,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= nearest.distance {
                    off[split_dim] = new_off;
                    self.nearest_one_recurse::<D>(
                        query,
                        further_node_idx,
                        next_split_dim,
                        nearest,
                        off,
                        rd,
                        level,
                        minor_level,
                        farther_leaf_idx,
                    );
                    off[split_dim] = old_off;
                }
            }

            #[allow(clippy::too_many_arguments)]
            #[cfg(not(feature = "modified_van_emde_boas"))]
            fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                stem_idx: u32,
                split_dim: usize,
                nearest: &mut NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
                mut level: usize,
                mut leaf_idx: u32,
            )
                where
                    D: DistanceMetric<A, K>,
            {
                if level > self.max_stem_level as usize || self.stems.is_empty() {
                    self.search_leaf_for_nearest_one::<D>(query, nearest, leaf_idx as usize);
                    return;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                let is_right_child = u32::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                leaf_idx <<= 1;
                let closer_leaf_idx = leaf_idx + is_right_child;
                let farther_leaf_idx = leaf_idx + (1 - is_right_child);

                let closer_node_idx = (stem_idx << 1) + is_right_child;
                let further_node_idx = (stem_idx << 1) + 1 - is_right_child;

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                level += 1;

                let next_split_dim = (split_dim + 1).rem(K);

                self.nearest_one_recurse::<D>(
                    query,
                    closer_node_idx,
                    next_split_dim,
                    nearest,
                    off,
                    rd,
                    level,
                    closer_leaf_idx,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= nearest.distance {
                    off[split_dim] = new_off;
                    self.nearest_one_recurse::<D>(
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
                }
            }

            #[inline]
            fn search_leaf_for_nearest_one<D>(
                &self,
                query: &[A; K],
                nearest: &mut NearestNeighbour<A, T>,
                leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
            {
                let leaf_slice = self.get_leaf_slice(leaf_idx);

                leaf_slice.nearest_one::<D>(
                    query,
                    &mut nearest.distance,
                    &mut nearest.item
                );
            }
        }
    };
}
