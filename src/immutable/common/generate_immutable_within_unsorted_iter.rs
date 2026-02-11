#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within_unsorted_iter {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn within_unsorted_iter<D>(
                &'a self,
                query: &'a [A; K],
                dist: A,
            ) -> WithinUnsortedIter<'a, A, T>
            where
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let stem_ordering = SO::new(stems_ptr);

                let gen = Gn::new_scoped(move |gen_scope| {
                    self.within_unsorted_iter_recurse::<D>(
                        query,
                        dist,
                        stem_ordering,
                        gen_scope,
                        &mut off,
                        A::zero(),
                    );

                    done!();
                });

                WithinUnsortedIter::new(gen)
            }

            #[allow(clippy::too_many_arguments)]
            fn within_unsorted_iter_recurse<'scope, D>(
                &'a self,
                query: &[A; K],
                radius: A,
                mut stem_ordering: SO,
                mut gen_scope: Scope<'scope, 'a, (), NearestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) -> Scope<'scope, 'a, (), NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                if stem_ordering.level() <= self.max_stem_level as usize {
                    let dim = stem_ordering.dim();
                    let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
                    let is_right_child: bool = *unsafe { query.get_unchecked(dim) } >= val;

                    let farther_so = stem_ordering.branch_relative(is_right_child);

                    let mut rd = rd;
                    let old_off = off[dim];
                    let new_off = query[dim].saturating_dist(val);

                    gen_scope = self.within_unsorted_iter_recurse::<D>(
                        query,
                        radius,
                        stem_ordering,
                        gen_scope,
                        off,
                        rd,
                    );

                    // Correct formula: rd_new = rd - old_off² + new_off²
                    let new_sq = D::dist1(new_off, A::default());
                    let old_sq = D::dist1(old_off, A::default());
                    rd = rd - old_sq + new_sq;

                    if rd <= radius {
                        off[dim] = new_off;
                        gen_scope = self.within_unsorted_iter_recurse::<D>(
                            query,
                            radius,
                            farther_so,
                            gen_scope,
                            off,
                            rd,
                        );
                        off[dim] = old_off;
                    }
                } else {
                    let leaf_slice = self.get_leaf_slice(leaf_idx);

                    leaf_slice
                        .content_items
                        .iter()
                        .enumerate()
                        .for_each(|(idx, &item)| {
                            let point = array_init::array_init(
                                |i| leaf_slice.content_points[i][idx]
                            );
                            let distance = D::dist(query, &point);

                            if distance <= radius {
                                gen_scope.yield_(NearestNeighbour {
                                    distance,
                                    item,
                                });
                            }
                        });
                }

                gen_scope
            }
        }
    };
}
