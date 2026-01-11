#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_best_n_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn best_n_within<D>(
                &self,
                query: &[A; K],
                dist: A,
                max_qty: NonZero<usize>,
            ) -> impl Iterator<Item = BestNeighbour<A, T>>
            where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                let mut off = [<A as AxisUnified>::zero(); K];
                let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::with_capacity(max_qty.into());

                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let stem_ordering = SO::new(stems_ptr);

                self.best_n_within_recurse::<D>(
                    query,
                    dist,
                    max_qty.into(),
                    stem_ordering,
                    &mut best_items,
                    &mut off,
                    <A as AxisUnified>::zero(),
                );

                best_items.into_iter()
            }

            #[allow(clippy::too_many_arguments)]
            fn best_n_within_recurse<D>(
                &self,
                query: &[A; K],
                radius: A,
                max_qty: usize,
                mut stem_ordering: SO,
                best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) where
                A: LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                if stem_ordering.level() > i32::from(self.max_stem_level) {
                    self.search_leaf_for_best_n_within::<D>(query, radius, max_qty, best_items, stem_ordering.leaf_idx());
                    return;
                }

                let dim = stem_ordering.dim();
                let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
                let is_right_child: bool = *unsafe { query.get_unchecked(dim) } >= val;

                let farther_so = stem_ordering.branch_relative(is_right_child);

                let mut rd = rd;
                let old_off = off[dim];
                let new_off = query[dim].saturating_dist(val);

                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
                    stem_ordering,
                    best_items,
                    off,
                    rd,
                );

                // Correct formula: rd_new = rd - old_off² + new_off²
                let new_sq = D::dist1(new_off, A::default());
                let old_sq = D::dist1(old_off, A::default());
                rd = rd - old_sq + new_sq;

                if rd <= radius {
                    off[dim] = new_off;

                    self.best_n_within_recurse::<D>(
                        query,
                        radius,
                        max_qty,
                        farther_so,
                        best_items,
                        off,
                        rd,
                    );

                    off[dim] = old_off;
                }
            }

            #[cfg_attr(not(feature = "no_inline"), inline)]
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
