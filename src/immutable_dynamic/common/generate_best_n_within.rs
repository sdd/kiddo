#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_best_n_within {
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
                A: BestFromDists<T>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::new();

                self.best_n_within_recurse::<D>(
                    query,
                    dist,
                    max_qty,
                    1,
                    &mut best_items,
                    &mut off,
                    A::zero(),
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
                best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) where
                A: BestFromDists<T>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                if stem_idx >= self.stems.len() {
                    self.leaf_best_n_within::<D>(
                        query,
                        radius,
                        max_qty,
                        best_items,
                        stem_idx - self.stems.len(),
                    );

                    return;
                }

                let left_child_idx = stem_idx << 1;

                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                self.prefetch_stems(left_child_idx);

                let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                // let val = self.stems[stem_idx];

                let split_dim = (*unsafe { self.split_dims.get_unchecked(stem_idx) }) as usize;

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
                // let is_left_child = usize::from(query[split_dim] < val);

                let closer_node_idx = left_child_idx + (1 - is_left_child);
                let further_node_idx = left_child_idx + is_left_child;

                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
                    closer_node_idx,
                    best_items,
                    off,
                    rd,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= radius {
                    off[split_dim] = new_off;
                    self.best_n_within_recurse::<D>(
                        query,
                        radius,
                        max_qty,
                        further_node_idx,
                        best_items,
                        off,
                        rd,
                    );
                    off[split_dim] = old_off;
                }
            }

            fn leaf_best_n_within<D>(
                &self,
                query: &[A; K],
                radius: A,
                max_qty: usize,
                best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
                leaf_idx: usize,
            ) where
                D: DistanceMetric<A, K>,
            {
                let leaf_offset = (*unsafe { self.leaf_offsets.get_unchecked(leaf_idx) }) as usize;
                let leaf_size = (*unsafe { self.leaf_sizes.get_unchecked(leaf_idx) }) as usize;

                let content_points: [&[A]; K] = init_array::init_array(|i| &unsafe {self.content_points.get_unchecked(i)}[leaf_offset..(leaf_offset+leaf_size)]);
                let content_items: &[T] = &self.content_items[leaf_offset..(leaf_offset+leaf_size)];

                let point_slice = crate::point_slice_ops_float::point_slice::PointSlice::new(content_points, content_items);

                point_slice.best_n_within::<D>(query, max_qty, radius, best_items);
            }
        }
    };
}
