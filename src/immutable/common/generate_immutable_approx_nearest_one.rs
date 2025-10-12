#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_approx_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
            where
                A: $crate::leaf_slice::float::LeafSliceFloat<T> + $crate::leaf_slice::float::LeafSliceFloatChunk<T, K>,
                D: DistanceMetric<A, K>,
                usize: Cast<T>,
            {
                let mut best_item = T::default();
                let mut best_dist = A::max_value();

                let leaf_idx = self.get_leaf_node_idx(query, None);

                #[cfg(feature = "rkyv_08")]
                #[allow(clippy::missing_transmute_annotations)]
                let leaf_slice = {
                    let leaf_extent = unsafe { self.leaf_extents.get_unchecked(leaf_idx) };
                    let start = Into::<u32>::into(leaf_extent.0);
                    let end = Into::<u32>::into(leaf_extent.1);

                    $crate::leaf_slice::float::LeafSlice::new(
                        array_init::array_init(|i| {
                            let archived_slice = &self.leaf_points[i][start as usize..end as usize];
                            unsafe { std::mem::transmute(archived_slice) }
                        }),
                        {
                            let archived_slice = &self.leaf_items[start as usize..end as usize];
                            unsafe { std::mem::transmute(archived_slice) }
                        }
                    )
                };

                #[cfg(not(feature = "rkyv_08"))]
                let leaf_slice = {
                    let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

                    $crate::leaf_slice::float::LeafSlice::new(
                        array_init::array_init(|i|
                            &self.leaf_points[i][start as usize..end as usize]
                        ),
                        &self.leaf_items[start as usize..end as usize],
                    )
                };

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
