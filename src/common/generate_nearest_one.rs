#[macro_export]
macro_rules! generate_nearest_one {
    ($kdtree:ident, $leafnode:ident, $comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
        where
            F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut off = [A::zero(); K];
        unsafe {
            self.nearest_one_recurse(
                query,
                distance_fn,
                self.root_index,
                0,
                T::zero(),
                A::max_value(),
                &mut off,
                A::zero(),
            )
        }
    }

    #[inline]
    unsafe fn nearest_one_recurse<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        mut best_item: T,
        mut best_dist: A,
        off: &mut [A; K],
        rd: A,
    ) -> (A, T)
        where
            F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

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

            let (dist, item) = self.nearest_one_recurse(
                query,
                distance_fn,
                closer_node_idx,
                next_split_dim,
                best_item,
                best_dist,
                off,
                rd,
            );

            if dist < best_dist {
                best_dist = dist;
                best_item = item;
            }

            // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
            //       so that updating rd is not hardcoded to sq euclidean
            rd = rd.rd_update(old_off, new_off);
            if rd <= best_dist {
                off[split_dim] = new_off;
                let (dist, item) = self.nearest_one_recurse(
                    query,
                    distance_fn,
                    further_node_idx,
                    next_split_dim,
                    best_item,
                    best_dist,
                    off,
                    rd,
                );
                off[split_dim] = old_off;

                if dist < best_dist {
                    best_dist = dist;
                    best_item = item;
                }
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            Self::search_content_for_best(
                query,
                distance_fn,
                &mut best_item,
                &mut best_dist,
                leaf_node,
            );
        }

        (best_dist, best_item)
    }

    fn search_content_for_best<F>(
        query: &[A; K],
        distance_fn: &F,
        best_item: &mut T,
        best_dist: &mut A,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size.az::<usize>())
            .for_each(|(idx, entry)| {
                let dist = distance_fn(query, entry);
                if dist < *best_dist {
                    *best_dist = dist;
                    *best_item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                }
            });
    }
}}}
