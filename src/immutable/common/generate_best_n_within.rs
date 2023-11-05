#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_best_n_within {
    ($comments:tt) => {
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
                A: BestFromDists<T, B>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                let mut off = [A::zero(); K];
                let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::new();

                self.best_n_within_recurse::<D>(
                    query,
                    dist,
                    max_qty,
                    1,
                    0,
                    &mut best_items,
                    &mut off,
                    A::zero(),
                );

                best_items.into_iter()
            }

            #[allow(clippy::too_many_arguments)]
            fn best_n_within_recurse<D>(
                &self,
                query: &[A; K],
                radius: A,
                max_qty: usize,
                stem_idx: usize,
                split_dim: usize,
                best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) where
                A: BestFromDists<T, B>,
                usize: Cast<T>,
                D: DistanceMetric<A, K>,
            {
                if stem_idx >= self.stems.len() {

                    #[cfg(feature = "no_unsafe")]
                    let leaf_node = &self.leaves[stem_idx - self.stems.len()];
                    #[cfg(not(feature = "no_unsafe"))]
                    let leaf_node = *unsafe { self.leaves.get_unchecked(stem_idx - self.stems.len()) };

                    let mut acc = [A::zero(); B];
                    (0..K).step_by(1).for_each(|dim| {
                        let qd = [query[dim]; B];

                        (0..leaf_node.size as usize).step_by(1).for_each(|idx| {
                            acc[idx] += D::dist1(leaf_node.content_points[dim][idx], qd[idx]);
                        });
                    });

                    acc
                        .iter()
                        .enumerate()
                        .take(leaf_node.size as usize)
                        .filter(|(_, &distance)| distance <= radius)
                        .for_each(|(idx, &distance)| {

                            #[cfg(not(feature = "no_unsafe"))]
                            let item = *unsafe { leaf_node.content_items.get_unchecked(idx) };
                            #[cfg(feature = "no_unsafe")]
                            let item = &leaf_node.content_items[idx];

                            if best_items.len() < max_qty {
                                best_items.push(BestNeighbour { distance, item });
                            } else {
                                let mut top = best_items.peek_mut().unwrap();
                                if item < top.item {
                                    top.item = item;
                                    top.distance = distance;
                                }
                            }
                        });

                    return;
                }

                let left_child_idx = stem_idx << 1;

                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                self.prefetch_stems(left_child_idx);

                #[cfg(not(feature = "no_unsafe"))]
                let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                #[cfg(feature = "no_unsafe")]
                let val = self.stems[stem_idx];

                let mut rd = rd;
                let old_off = off[split_dim];
                let new_off = query[split_dim].saturating_dist(val);

                #[cfg(not(feature = "no_unsafe"))]
                let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
                #[cfg(feature = "no_unsafe")]
                let is_left_child = usize::from(query[split_dim] < val);

                let closer_node_idx = left_child_idx + (1 - is_left_child);
                let further_node_idx = left_child_idx + is_left_child;

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
            }
        }
    };
}
