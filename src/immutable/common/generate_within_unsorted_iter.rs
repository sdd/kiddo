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
                    unsafe {
                        self.within_unsorted_iter_recurse::<D>(
                            query,
                            dist,
                            1,
                            0,
                            gen_scope,
                            &mut off,
                            A::zero(),
                        );
                    }

                    done!();
                });

                WithinUnsortedIter::new(gen)
            }

            #[allow(clippy::too_many_arguments)]
            fn within_unsorted_iter_recurse<D>(
                &'a self,
                query: &[A; K],
                radius: A,
                stem_idx: usize,
                split_dim: usize,
                mut gen_scope: Scope<'a, (), NearestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) -> Scope<(), NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
                if stem_idx < self.stems.len() {
                    let left_child_idx = stem_idx << 1;
                    self.prefetch_stems(left_child_idx);

                    let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                    // let val = self.stems[stem_idx];

                    let mut rd = rd;
                    let old_off = off[split_dim];
                    let new_off = query[split_dim].saturating_dist(val);

                    let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
                    // let is_left_child = usize::from(query[split_dim] < val);

                    let closer_node_idx = left_child_idx + (1 - is_left_child);
                    let further_node_idx = left_child_idx + is_left_child;

                    let next_split_dim = (split_dim + 1).rem(K);

                    gen_scope = self.within_unsorted_iter_recurse::<D>(
                        query,
                        radius,
                        closer_node_idx,
                        next_split_dim,
                        gen_scope,
                        off,
                        rd,
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
                        );
                        off[split_dim] = old_off;
                    }
                } else {
                    let leaf_node = self
                        .leaves
                        .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

                    leaf_node
                        .content_points
                        .iter()
                        .enumerate()
                        .take(leaf_node.size.az::<usize>())
                        .for_each(|(idx, entry)| {
                            let distance = D::dist(query, entry);

                            if distance < radius {
                                gen_scope.yield_(NearestNeighbour {
                                    distance,
                                    item: *leaf_node.content_items.get_unchecked(idx.az::<usize>()),
                                });
                            }
                        });
                }

                gen_scope
            }
        }
    };
}
