#[macro_export]
macro_rules! generate_immutable_nearest_n {
    ($comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn nearest_n<D>(&self, query: &[A; K], qty: usize) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut result: BinaryHeap<NearestNeighbour<A, T>> = BinaryHeap::with_capacity(qty);

        self.nearest_n_recurse::<D>(
            query,
            1,
            0,
            &mut result,
            &mut off,
            A::zero(),
        );

        result.into_sorted_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn nearest_n_recurse<D>(
        &self,
        query: &[A; K],
        stem_idx: usize,
        split_dim: usize,
        results: &mut BinaryHeap<NearestNeighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
        D: DistanceMetric<A, K>,
    {
        if stem_idx >= self.stems.len() {
            let leaf_node = &self.leaves[stem_idx - self.stems.len()];

            leaf_node
                .content_points
                .iter()
                .take(leaf_node.size as usize)
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: A = D::dist(query, entry);
                    if Self::dist_belongs_in_heap(distance, results) {
                        let item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        let element = NearestNeighbour { distance, item };
                        if results.len() < results.capacity() {
                            results.push(element)
                        } else {
                            let mut top = results.peek_mut().unwrap();
                            if element.distance < top.distance {
                                *top = element;
                            }
                        }
                    }
                });

            return;
        }

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

        self.nearest_n_recurse::<D>(query, closer_node_idx, next_split_dim, results, off, rd);

        rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

        if Self::dist_belongs_in_heap(rd, results) {
            off[split_dim] = new_off;
            self.nearest_n_recurse::<D>(query, further_node_idx, next_split_dim, results, off, rd);
            off[split_dim] = old_off;
        }
    }

    fn dist_belongs_in_heap(dist: A, heap: &BinaryHeap<NearestNeighbour<A, T>>) -> bool {
        heap.is_empty() || dist < heap.peek().unwrap().distance || heap.len() < heap.capacity()
    }
}}}
