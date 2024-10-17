#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_approx_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
            where
                A: BestFromDists<T>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,
            {
                let mut stem_idx = 1;
                let mut best_dist: A = A::infinity();
                let mut best_item: T = T::zero();
                let stem_len = self.stems.len();

                while stem_idx < stem_len {
                    let left_child_idx = stem_idx << 1;

                    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                    self.prefetch_stems(left_child_idx);

                    let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                    let split_dim = (*unsafe { self.split_dims.get_unchecked(stem_idx) }) as usize;
                    let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim) } >= val);

                    stem_idx = left_child_idx + is_right_child;
                }

                let leaf_offset = (*unsafe { self.leaf_offsets.get_unchecked(stem_idx - stem_len) }) as usize;
                let leaf_size = (*unsafe { self.leaf_sizes.get_unchecked(stem_idx - stem_len) }) as usize;

                let content_points: [&[A]; K] = init_array::init_array(|i| &unsafe {self.content_points.get_unchecked(i)}[leaf_offset..(leaf_offset+leaf_size)]);
                let content_items: &[T] = &self.content_items[leaf_offset..(leaf_offset+leaf_size)];

                let point_slice = crate::point_slice_ops_float::point_slice::PointSlice::new(content_points, content_items);

                point_slice.nearest_one::<D>(query, &mut best_dist, &mut best_item);

                NearestNeighbour {
                    distance: best_dist,
                    item: best_item,
                }
            }
        }
    };
}
