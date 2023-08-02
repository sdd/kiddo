use crate::float::kdtree::Axis;
use crate::immutable_float::kdtree::ImmutableKdTree;

use crate::types::Content;
use std::collections::BinaryHeap;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize>
    ImmutableKdTree<A, T, K, B>
{
    /// Finds the "best" `n` elements within `dist` of `query`.
    ///
    /// Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt).
    /// Returns an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable_float::kdtree::ImmutableKdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let content: Vec<[f64; 3]> = vec!(
    ///     [1.0, 2.0, 5.0],
    ///     [2.0, 3.0, 6.0],
    ///     [200.0, 300.0, 600.0],
    /// );
    ///
    /// let mut tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::optimized_from(&content);
    ///
    /// let mut best_n_within = tree.best_n_within(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean);
    /// let first = best_n_within.next().unwrap();
    ///
    /// assert_eq!(first, 1);
    /// ```
    #[inline]
    pub fn best_n_within<F>(
        &self,
        query: &[A; K],
        dist: A,
        max_qty: usize,
        distance_fn: &F,
    ) -> impl Iterator<Item = T>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut off = [A::zero(); K];
        let mut best_items: BinaryHeap<T> = BinaryHeap::new();

        self.best_n_within_recurse(
            query,
            dist,
            max_qty,
            distance_fn,
            1,
            0,
            &mut best_items,
            &mut off,
            A::zero(),
        );


        best_items.into_iter()
    }

    #[allow(clippy::too_many_arguments)]
    fn best_n_within_recurse<F>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
        stem_idx: usize,
        split_dim: usize,
        best_items: &mut BinaryHeap<T>,
        off: &mut [A; K],
        rd: A,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if stem_idx >= self.stems.len() {
            let leaf_node = &self.leaves[stem_idx - self.stems.len()];

            leaf_node
                .content_points
                .iter()
                .take(leaf_node.size)
                .map(|entry| distance_fn(query, entry))
                .enumerate()
                .filter(|(_, distance)| *distance <= radius)
                .for_each(|(idx, _)| {
                    let item = * unsafe { leaf_node.content_items.get_unchecked(idx) };
                    if best_items.len() < max_qty {
                        best_items.push(item);
                    } else {
                        let mut top = best_items.peek_mut().unwrap();
                        if item < *top {
                            *top = item;
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
        let new_off = query[split_dim] - val;

        let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
        // let is_left_child = usize::from(query[split_dim] < val);

        let closer_node_idx = left_child_idx + (1 - is_left_child);
        let further_node_idx = left_child_idx + is_left_child;

        let next_split_dim = (split_dim + 1).rem(K);

        self.best_n_within_recurse(
            query,
            radius,
            max_qty,
            distance_fn,
            closer_node_idx,
            next_split_dim,
            best_items,
            off,
            rd,
        );

        // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
        //       so that updating rd is not hardcoded to sq euclidean
        rd = rd + new_off * new_off - old_off * old_off;

        if rd <= radius {
            off[split_dim] = new_off;
            self.best_n_within_recurse(
                query,
                radius,
                max_qty,
                distance_fn,
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
    use crate::float::distance::squared_euclidean;
    use crate::immutable_float::kdtree::ImmutableKdTree;
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

        let tree: ImmutableKdTree<AX, u32, 2, 4> = ImmutableKdTree::optimize_from(&content_to_add);

        assert_eq!(tree.size(), 16);

        let query = [9f64, 0f64];
        let radius = 20000f64;
        let max_qty = 3;
        let expected = vec![14, 0, 9];

        let result: Vec<_> = tree
            .best_n_within(&query, radius, max_qty, &squared_euclidean)
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
            println!("{}, {}", query[0].to_string(), query[1].to_string());

            let result: Vec<_> = tree
                .best_n_within(&query, radius, max_qty, &squared_euclidean)
                .collect();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = 2;

        let content_to_add: Vec<[AX; 2]> = (0..TREE_SIZE)
            .map(|_| rand::random::<[AX; 2]>())
            .collect();

        let tree: ImmutableKdTree<AX, u32, 2, 32> = ImmutableKdTree::optimize_from(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[AX; 2]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[AX; 2]>())
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty);

            let result: Vec<_> = tree
                .best_n_within(&query_point, radius, max_qty, &squared_euclidean)
                .collect();
            assert_eq!(result, expected);
        }
    }

    fn linear_search(
        content: &[[f64; 2]],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<u32> {
        let mut best_items = Vec::with_capacity(max_qty);

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean(query, &p);
            if dist <= radius {
                if best_items.len() < max_qty {
                    best_items.push(idx as u32);
                } else {
                    if (idx as u32) < *best_items.last().unwrap() {
                        best_items.pop().unwrap();
                        best_items.push(idx as u32);
                    }
                }
            }
            best_items.sort_unstable();
        }
        best_items.reverse();

        best_items
    }
}
