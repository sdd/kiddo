use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;

use crate::best_neighbour::BestNeighbour;
use crate::distance_metric::DistanceMetric;
use crate::types::Content;
use std::collections::BinaryHeap;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Finds the "best" `n` elements within `dist` of `query`.
    ///
    /// Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt).
    /// Returns an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    /// use kiddo::float::distance::SquaredEuclidean;
    ///
    /// let content: Vec<[f64; 3]> = vec!(
    ///     [1.0, 2.0, 5.0],
    ///     [2.0, 3.0, 6.0],
    ///     [200.0, 300.0, 600.0],
    /// );
    ///
    /// let mut tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);
    ///
    /// let mut best_n_within = tree.best_n_within::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64, 1);
    /// let first = best_n_within.next().unwrap();
    ///
    /// assert_eq!(first.item, 0);
    /// ```
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
        D: DistanceMetric<A, K>,
    {
        if stem_idx >= self.stems.len() {
            let leaf_node = &self.leaves[stem_idx - self.stems.len()];

            leaf_node
                .content_points
                .iter()
                .take(leaf_node.size)
                .map(|entry| D::dist(query, entry))
                .enumerate()
                .filter(|(_, distance)| *distance <= radius)
                .for_each(|(idx, distance)| {
                    let item = *unsafe { leaf_node.content_items.get_unchecked(idx) };
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

#[cfg(test)]
mod tests {
    use crate::best_neighbour::BestNeighbour;
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::SquaredEuclidean;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use rand::Rng;

    type AX = f64;

    #[test]
    fn can_query_best_n_items_within_radius() {
        let content_to_add = [
            [9f64, 0f64],
            [4f64, 500f64],
            [12f64, -300f64],
            [7f64, 200f64],
            [13f64, -400f64],
            [6f64, 300f64],
            [2f64, 700f64],
            [14f64, -500f64],
            [3f64, 600f64],
            [10f64, -100f64],
            [16f64, -700f64],
            [1f64, 800f64],
            [15f64, -600f64],
            [5f64, 400f64],
            [8f64, 100f64],
            [11f64, -200f64],
        ];

        let tree: ImmutableKdTree<AX, i32, 2, 4> = ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), 16);

        let query = [9f64, 0f64];
        let radius = 20000f64;
        let max_qty = 3;
        let expected = vec![
            BestNeighbour {
                distance: 10001.0,
                item: 14,
            },
            BestNeighbour {
                distance: 0.0,
                item: 0,
            },
            BestNeighbour {
                distance: 10001.0,
                item: 9,
            },
        ];

        let result: Vec<_> = tree
            .best_n_within::<SquaredEuclidean>(&query, radius, max_qty)
            .collect();
        assert_eq!(result, expected);

        let max_qty = 2;

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query = [
                rng.gen_range(-10f64..20f64),
                rng.gen_range(-1000f64..1000f64),
            ];
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query, radius, max_qty);
            //println!("{}, {}", query[0].to_string(), query[1].to_string());

            let result: Vec<_> = tree
                .best_n_within::<SquaredEuclidean>(&query, radius, max_qty)
                .collect();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = 2;

        let content_to_add: Vec<[AX; 2]> =
            (0..TREE_SIZE).map(|_| rand::random::<[AX; 2]>()).collect();

        let tree: ImmutableKdTree<AX, i32, 2, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[AX; 2]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[AX; 2]>())
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty);

            let result: Vec<_> = tree
                .best_n_within::<SquaredEuclidean>(&query_point, radius, max_qty)
                .collect();
            assert_eq!(result, expected);
        }
    }

    fn linear_search(
        content: &[[f64; 2]],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<BestNeighbour<f64, i32>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for (item, p) in content.iter().enumerate() {
            let distance = SquaredEuclidean::dist(query, &p);
            if distance <= radius {
                if best_items.len() < max_qty {
                    best_items.push(BestNeighbour {
                        distance,
                        item: item as i32,
                    });
                } else {
                    if (item as i32) < (*best_items.last().unwrap()).item {
                        best_items.pop().unwrap();
                        best_items.push(BestNeighbour {
                            distance,
                            item: item as i32,
                        });
                    }
                }
            }
            best_items.sort_unstable();
        }
        best_items.reverse();

        best_items
    }
}
