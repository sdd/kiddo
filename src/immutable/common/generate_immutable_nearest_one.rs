#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_nearest_one {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            // #[cfg_attr(not(feature = "no_inline"), inline)]
            #[inline(never)]
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

                // Add a marker for LLVM-MCA to mark the start of the block we want to analyze
                unsafe {
                    core::arch::asm!("# LLVM-MCA-BEGIN");
                }

                self.nearest_one_recurse::<D>(
                    query,
                    SO::new(),
                    &mut result,
                    &mut off,
                    A::zero(),
                );

                // LLVM-MCA end marker
                unsafe {
                     core::arch::asm!("# LLVM-MCA-END");
                }

                result
            }

            // #[cfg_attr(not(feature = "no_inline"), inline)]
            // #[inline(never)]
            pub fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                mut stem_ordering: SO,
                nearest: &mut NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
            )
                where
                    D: DistanceMetric<A, K>,
            {
                if stem_ordering.level() > Into::<i32>::into(self.max_stem_level) || self.stems.is_empty() {
                    self.search_leaf_for_nearest_one::<D>(query, nearest, stem_ordering.leaf_idx());
                    return;
                }

                let dim = stem_ordering.dim();
                let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
                let is_right_child = *unsafe { query.get_unchecked(dim) } >= val;

                let farther_so = stem_ordering.branch_relative(is_right_child);

                let mut rd = rd;
                let old_off = off[dim];
                let new_off = query[dim].saturating_dist(val);

                self.nearest_one_recurse::<D>(
                    query,
                    stem_ordering,
                    nearest,
                    off,
                    rd,
                );

                rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                if rd <= nearest.distance {
                    off[dim] = new_off;
                    self.nearest_one_recurse::<D>(
                        query,
                        farther_so,
                        nearest,
                        off,
                        rd,
                    );
                    off[dim] = old_off;
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
