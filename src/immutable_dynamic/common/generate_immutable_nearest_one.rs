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
                    1,
                    NearestNeighbour {
                        distance: A::max_value(),
                        item: T::zero(),
                    },
                    &mut off,
                    A::zero(),
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                stem_idx: usize,
                mut nearest: NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
            ) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                if stem_idx >= self.stems.len() {
                    self.search_leaf_for_nearest::<D>(query, &mut nearest, stem_idx - self.stems.len());
                    return nearest;
                }

                let left_child_idx = stem_idx << 1;

                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                self.prefetch_stems(left_child_idx);

                // let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                let val = self.stems[stem_idx];
                let split_dim = self.split_dims[stem_idx] as usize;

                let mut rd = rd;
                let old_off = off[split_dim];
                // let new_off = (query[split_dim] * query[split_dim]) - (val * val);
                let new_off = query[split_dim].saturating_dist(val);
                // let new_off = query[split_dim] - val;

                let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
                // let is_left_child = usize::from(query[split_dim] < val);

                let closer_node_idx = left_child_idx + (1 - is_left_child);
                let further_node_idx = left_child_idx + is_left_child;

                let nearest_neighbour =
                    self.nearest_one_recurse::<D>(query, closer_node_idx, nearest, off, rd);

                if nearest_neighbour < nearest {
                    nearest = nearest_neighbour;
                }

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= nearest.distance {
                    off[split_dim] = new_off;
                    let result = self.nearest_one_recurse::<D>(
                        query,
                        further_node_idx,
                        nearest,
                        off,
                        rd,
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
                let leaf_offset = (*unsafe { self.leaf_offsets.get_unchecked(leaf_idx) }) as usize;
                let leaf_size = (*unsafe { self.leaf_sizes.get_unchecked(leaf_idx) }) as usize;

                let content_points: [&[A]; K] = init_array::init_array(|i| &unsafe {self.content_points.get_unchecked(i)}[leaf_offset..(leaf_offset+leaf_size)]);
                let content_items: &[T] = &self.content_items[leaf_offset..(leaf_offset+leaf_size)];

                let point_slice = crate::point_slice_ops_float::point_slice::PointSlice::new(content_points, content_items);

                let mut best_item = nearest.item;
                let mut best_dist = nearest.distance;

                point_slice.nearest_one::<D>(query, &mut best_dist, &mut best_item);

                nearest.distance = best_dist;
                nearest.item = best_item;
            }
        }
    };
}
