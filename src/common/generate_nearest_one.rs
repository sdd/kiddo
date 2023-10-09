#[doc(hidden)]
#[macro_export]
macro_rules! generate_nearest_one {
    ($leafnode:ident, $comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];

                unsafe {
                    self.nearest_one_recurse::<D>(
                        query,
                        self.root_index,
                        0,
                        NearestNeighbour { distance: A::max_value(), item: T::zero() },
                        &mut off,
                        A::zero(),
                    )
                }
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                curr_node_idx: IDX,
                split_dim: usize,
                mut nearest: NearestNeighbour<A, T>,
                off: &mut [A; K],
                rd: A,
            ) -> NearestNeighbour<A, T>
                where
                    D: DistanceMetric<A, K>,
            {
                if is_stem_index(curr_node_idx) {
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

                    let nearest_neighbour = self.nearest_one_recurse::<D>(
                        query,
                        closer_node_idx,
                        next_split_dim,
                        nearest,
                        off,
                        rd,
                    );

                    if nearest_neighbour < nearest {
                        nearest = nearest_neighbour;
                    }

                    rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

                    if rd <= nearest.distance {
                        off[split_dim] = new_off;
                        let result = self.nearest_one_recurse::<D>(
                            query,
                            further_node_idx,
                            next_split_dim,
                            nearest,
                            off,
                            rd,
                        );
                        off[split_dim] = old_off;

                        if result < nearest {
                            nearest = result;
                        }
                    }
                } else {
                    let leaf_node = self
                        .leaves
                        .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

                    Self::search_content_for_nearest::<D>(
                        query,
                        &mut nearest,
                        leaf_node,
                    );
                }

                nearest
            }

            #[inline]
            fn search_content_for_nearest<D>(
                query: &[A; K],
                nearest: &mut NearestNeighbour<A, T>,
                leaf_node: &$leafnode<A, T, K, B, IDX>,
            ) where
                D: DistanceMetric<A, K>,
            {
                leaf_node
                    .content_points
                    .iter()
                    .enumerate()
                    .take(leaf_node.size.az::<usize>())
                    .for_each(|(idx, entry)| {
                        let dist = D::dist(query, entry);
                        if dist < nearest.distance {
                            nearest.distance = dist;
                            nearest.item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        }
                    });
            }
        }
    };
}
