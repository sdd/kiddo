#[macro_export]
macro_rules! generate_within_unsorted_iter {
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
                    self.root_index,
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
    unsafe fn within_unsorted_iter_recurse<D>(
        &'a self,
        query: &[A; K],
        radius: A,
        curr_node_idx: IDX,
        split_dim: usize,
        mut gen_scope: Scope<'a, (), NearestNeighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) -> Scope<(), NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        if is_stem_index(curr_node_idx) {
            let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim].saturating_dist(node.split_val);

            let [closer_node_idx, further_node_idx] =
                if *query.get_unchecked(split_dim) < node.split_val {
                    [node.left, node.right]
                } else {
                    [node.right, node.left]
                };
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

            rd += D::dist1(old_off, new_off);

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
}}}
