#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_dynamic_approx_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
            where
                A: $crate::float_leaf_slice::leaf_slice::LeafSliceFloat<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,
            {
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2;

                let mut curr_idx: usize = 0;
                let mut dim: usize = 0;
                let mut best_item = T::zero();
                let mut best_dist = A::max_value();
                let mut level: usize = 0;
                let mut leaf_idx: usize = 0;

                while level <= self.max_stem_level {
                    let val = *unsafe { self.stems.get_unchecked(curr_idx) };
                    let is_right_child = *unsafe { query.get_unchecked(dim) } >= val;

                    curr_idx = modified_van_emde_boas_get_child_idx_v2(curr_idx, is_right_child, level);
                    let is_right_child = usize::from(is_right_child);
                    leaf_idx = (leaf_idx << 1) + is_right_child;

                    level += 1;
                    dim = (dim + 1) % K;
                }

                let leaf_extent: core::range::Range<usize> = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

                let leaf_slice = $crate::float_leaf_slice::leaf_slice::LeafSlice::new(
                    array_init::array_init(|i|
                        &self.leaf_points[i][leaf_extent]
                    ),
                    &self.leaf_items[leaf_extent],
                );

                leaf_slice.nearest_one::<D>(
                    query,
                    &mut best_dist,
                    &mut best_item
                );

                NearestNeighbour {
                    distance: best_dist,
                    item: best_item,
                }
            }
        }
    };
}
