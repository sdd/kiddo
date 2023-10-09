#[doc(hidden)]
#[macro_export]
macro_rules! generate_best_n_within {
    ($leafnode:ident, $comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_qty: usize,
    ) -> impl Iterator<Item = BestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::new();

        unsafe {
            self.best_n_within_recurse::<D>(
                query,
                dist,
                max_qty,
                self.root_index,
                0,
                &mut best_items,
                &mut off,
                A::zero(),
            );
        }

        best_items.into_iter()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn best_n_within_recurse<D>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        curr_node_idx: IDX,
        split_dim: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
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

            self.best_n_within_recurse::<D>(
                query,
                radius,
                max_qty,
                closer_node_idx,
                next_split_dim,
                best_items,
                off,
                rd,
            );

            rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

            if rd <= radius {
                off[split_dim] = new_off;
                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
                    further_node_idx,
                    next_split_dim,
                    best_items,
                    off,
                    rd,
                );
                off[split_dim] = old_off;
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            Self::process_leaf_node::<D>(query, radius, max_qty, best_items, leaf_node);
        }
    }

    #[inline]
    unsafe fn process_leaf_node<D>(
        query: &[A; K],
        radius: A,
        max_qty: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
    ) where
        D: DistanceMetric<A, K>,
    {
        leaf_node
            .content_points
            .iter()
            .take(leaf_node.size.az::<usize>())
            .map(|entry| D::dist(query, entry))
            .enumerate()
            .filter(|(_, distance)| *distance <= radius)
            .for_each(|(idx, distance)| {
                Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx, distance)
            });
    }

    #[inline]
    unsafe fn get_item_and_add_if_good(
        max_qty: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
        idx: usize,
        distance: A,
    ) {
        let item = *leaf_node.content_items.get_unchecked(idx.az::<usize>());
        if best_items.len() < max_qty {
            best_items.push(BestNeighbour{ distance, item });
        } else {
            let mut top = best_items.peek_mut().unwrap();
            if item < top.item {
                top.item = item;
                top.distance = distance;
            }
        }
    }
}}}
