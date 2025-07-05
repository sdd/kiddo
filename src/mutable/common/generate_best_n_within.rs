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
        let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::with_capacity(max_qty);
        let root_index: IDX = *transform(&self.root_index);

        unsafe {
            self.best_n_within_recurse::<D>(
                query,
                dist,
                max_qty,
                root_index,
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
            let split_val: A = *transform(&node.split_val);
            let node_left: IDX = *transform(&node.left);
            let node_right: IDX = *transform(&node.right);

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim].saturating_dist(split_val);

            let [closer_node_idx, further_node_idx] =
                if *query.get_unchecked(split_dim) < split_val {
                    [node_left, node_right]
                } else {
                    [node_right, node_left]
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
        let size: IDX = *transform(&leaf_node.size);

        leaf_node
            .content_points
            .iter()
            .take(size.az::<usize>())
            .map(|entry| D::dist(query, transform(entry)))
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
        let item = leaf_node.content_items.get_unchecked(idx.az::<usize>());
        let item = *transform(item);

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
