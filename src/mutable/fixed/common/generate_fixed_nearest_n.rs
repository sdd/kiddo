#[doc(hidden)]
#[macro_export]
macro_rules! generate_fixed_nearest_n {
    ($comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn nearest_n<D, R: AxisFixed>(&self, query: &[A; K], qty: usize) -> Vec<NearestNeighbour<R, T>>
    where
        D: DistanceMetricFixed<A, K, R>,
    {
        let mut off = [A::zero(); K];
        let mut result: BinaryHeap<NearestNeighbour<R, T>> = BinaryHeap::with_capacity(qty);
        let root_index: IDX = *transform(&self.root_index);

        unsafe {
            self.nearest_n_recurse::<D, R>(
                query,
                root_index,
                0,
                &mut result,
                &mut off,
                R::zero(),
            )
        }

        result.into_sorted_vec()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn nearest_n_recurse<D, R: AxisFixed>(
        &self,
        query: &[A; K],
        curr_node_idx: IDX,
        split_dim: usize,
        results: &mut BinaryHeap<NearestNeighbour<R, T>>,
        off: &mut [A; K],
        rd: R,
    ) where
        D: DistanceMetricFixed<A, K, R>,
    {
        if is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());
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

            self.nearest_n_recurse::<D, R>(
                query,
                closer_node_idx,
                next_split_dim,
                results,
                off,
                rd,
            );

            rd = AxisFixed::rd_update(rd, D::dist1(new_off, old_off));

            if Self::dist_belongs_in_heap(rd, results) {
                off[split_dim] = new_off;
                self.nearest_n_recurse::<D, R>(
                    query,
                    further_node_idx,
                    next_split_dim,
                    results,
                    off,
                    rd,
                );
                off[split_dim] = old_off;
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            let size: IDX = *transform(&leaf_node.size);

            leaf_node
                .content_points
                .iter()
                .take(size.az::<usize>())
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: R = D::dist(query, transform(entry));

                    if Self::dist_belongs_in_heap::<R>(distance, results) {
                        let item = unsafe { leaf_node.content_items.get_unchecked(idx) };
                        let item = *transform(item);

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
        }
    }

    #[inline]
    fn dist_belongs_in_heap<R: AxisFixed>(dist: R, heap: &BinaryHeap<NearestNeighbour<R, T>>) -> bool {
        heap.is_empty() || dist < heap.peek().unwrap().distance || heap.len() < heap.capacity()
    }
}}}
