#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut result = NearestNeighbour {
                    distance: A::max_value(),
                    item: T::default(),
                };

                if self.stems.is_empty() {
                    self.search_leaf_for_nearest_one::<D>(query, &mut result, 0);
                    return result;
                }

                let stem_ordering = SO::new_query();
                let initial_stem_idx: usize = SO::get_initial_idx();

                self.nearest_one_recurse::<D>(
                    query,
                    initial_stem_idx,
                    stem_ordering,
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
            #[cfg_attr(not(feature = "no_inline"), inline)]
            fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                stem_idx: usize,
                mut stem_ordering: SO,
                split_dim: u64,
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

                if level > Into::<i32>::into(self.max_stem_level) || self.stems.is_empty() {
                    self.search_leaf_for_nearest_one::<D>(query, nearest, leaf_idx as usize);
                    return;
                }

                let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                let is_right_child: bool = *unsafe { query.get_unchecked(split_dim as usize) } >= val;

                let (closer_node_idx, further_node_idx) = stem_ordering.get_closer_and_further_child_idx(stem_idx, is_right_child);

                leaf_idx <<= 1;
                let is_right_child = u32::from(is_right_child);
                let closer_leaf_idx = leaf_idx + is_right_child;
                let farther_leaf_idx = leaf_idx + (1 - is_right_child);


                let mut rd = rd;
                let old_off = off[split_dim as usize];
                let new_off = query[split_dim as usize].saturating_dist(val);

                level += 1;
                minor_level += 1;
                minor_level.cmovnz(&0, u8::from(minor_level == 3));

                let mut next_split_dim = split_dim + 1;
                next_split_dim.cmovnz(&0, u8::from(next_split_dim == K as u64));

                self.nearest_one_recurse::<D>(
                    query,
                    closer_node_idx,
                    stem_ordering.clone(),
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
                    off[split_dim as usize] = new_off;
                    self.nearest_one_recurse::<D>(
                        query,
                        further_node_idx,
                        stem_ordering,
                        next_split_dim,
                        nearest,
                        off,
                        rd,
                        level,
                        minor_level,
                        farther_leaf_idx,
                    );
                    off[split_dim as usize] = old_off;
                }
            }

            #[cfg_attr(not(feature = "no_inline"), inline)]
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
