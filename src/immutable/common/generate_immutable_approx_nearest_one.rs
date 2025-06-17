#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_approx_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
            where
                A: $crate::float_leaf_slice::leaf_slice::LeafSliceFloat<T> + $crate::float_leaf_slice::leaf_slice::LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,
            {
                #[cfg(feature = "modified_van_emde_boas")]
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2;

                #[cfg(feature = "modified_van_emde_boas")]
                let mut curr_idx: usize = 0;
                #[cfg(not(feature = "modified_van_emde_boas"))]
                let mut curr_idx: usize = 1;

                let mut dim: usize = 0;
                let mut best_item = T::zero();
                let mut best_dist = A::max_value();
                let mut level: i32 = 0;
                let mut leaf_idx: usize = 0;

                while level <= Into::<i32>::into(self.max_stem_level) {
                    let val = *unsafe { self.stems.get_unchecked(curr_idx) };
                    let is_right_child = *unsafe { query.get_unchecked(dim) } >= val;

                    #[cfg(feature = "modified_van_emde_boas")]
                    let next_idx = modified_van_emde_boas_get_child_idx_v2(curr_idx as u32, is_right_child, level as u32) as usize;
                    #[cfg(not(feature = "modified_van_emde_boas"))]
                    let next_idx = (curr_idx << 1) + usize::from(is_right_child);

                    curr_idx = next_idx;

                    let is_right_child = usize::from(is_right_child);
                    leaf_idx = (leaf_idx << 1) + is_right_child;

                    level += 1;
                    dim = (dim + 1) % K;
                }

                // Handle the archived tuple differently depending on the feature
                #[cfg(feature = "rkyv_08")]
                let leaf_extent = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
                #[cfg(feature = "rkyv_08")]
                let start = leaf_extent.0;
                #[cfg(feature = "rkyv_08")]
                let end = leaf_extent.1;

                #[cfg(not(feature = "rkyv_08"))]
                let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

                let start = Into::<u32>::into(start);
                let end = Into::<u32>::into(end);

                let leaf_slice = $crate::float_leaf_slice::leaf_slice::LeafSlice::new(
                    array_init::array_init(|i|
                        #[cfg(feature = "rkyv_08")]
                        {
                            // Transmute the archived slice to the regular type slice
                            let archived_slice = &self.leaf_points[i][start as usize..end as usize];
                            #[allow(clippy::missing_transmute_annotations)]
                            unsafe { std::mem::transmute(archived_slice) }
                        },
                        #[cfg(not(feature = "rkyv_08"))]
                        &self.leaf_points[i][start as usize..end as usize]
                    ),
                    #[cfg(feature = "rkyv_08")]
                    {
                        // Transmute the archived slice to the regular type slice
                        let archived_slice = &self.leaf_items[start as usize..end as usize];
                        #[allow(clippy::missing_transmute_annotations)]
                        unsafe { std::mem::transmute(archived_slice) }
                    },
                    #[cfg(not(feature = "rkyv_08"))]
                    &self.leaf_items[start as usize..end as usize],
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
