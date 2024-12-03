#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_within_unsorted_iter {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted_iter<D>(
                &'a self,
                query: &'a [A; K],
                dist: A,
            ) -> WithinUnsortedIter<'a, A, T>
            where
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];

                let gen = Gn::new_scoped(move |gen_scope| {
                    self.within_unsorted_iter_recurse::<D>(
                        query,
                        dist,
                        0,
                        0,
                        gen_scope,
                        &mut off,
                        A::zero(),
                        0,
                        0,
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
                stem_idx: usize,
                split_dim: usize,
                mut gen_scope: Scope<'scope, 'a, (), NearestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
                mut level: usize,
                mut leaf_idx: usize,
            ) -> Scope<'scope, 'a, (), NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                use $crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;

                if level <= self.max_stem_level as usize {
                    let val = *unsafe { self.stems.get_unchecked(stem_idx as usize) };
                    let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim as usize) } >= val);

                    leaf_idx <<= 1;
                    let closer_leaf_idx = leaf_idx + is_right_child;
                    let further_leaf_idx = leaf_idx + (1 - is_right_child);

                    let closer_node_idx = modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx as u32, is_right_child == 1, /*minor_*/level as u32) as usize;
                    let further_node_idx =  modified_van_emde_boas_get_child_idx_v2_branchless(stem_idx as u32, is_right_child == 0, /*minor_*/level as u32) as usize;

                    let mut rd = rd;
                    let old_off = off[split_dim];
                    let new_off = query[split_dim].saturating_dist(val);

                    level += 1;
                    let next_split_dim = (split_dim + 1).rem(K);
                    // minor_level += 1;
                    // minor_level.cmovnz(&0, u8::from(minor_level == 3));

                    gen_scope = self.within_unsorted_iter_recurse::<D>(
                        query,
                        radius,
                        closer_node_idx,
                        next_split_dim,
                        gen_scope,
                        off,
                        rd,
                        level,
                        closer_leaf_idx,
                    );

                    rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                    if rd <= radius {
                        off[split_dim] = new_off;
                        gen_scope = self.within_unsorted_iter_recurse::<D>(
                            query,
                            radius,
                            further_node_idx,
                            next_split_dim,
                            gen_scope,
                            off,
                            rd,
                            level,
                            further_leaf_idx,
                        );
                        off[split_dim] = old_off;
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

                            if distance < radius {
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
