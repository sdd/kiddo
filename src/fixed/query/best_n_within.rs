use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::fixed::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{is_stem_index, Content, Index};

use crate::generate_best_n_within;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_best_n_within!(
        LeafNode,
        (r#"Queries the tree to find the best `n` elements within `dist` of `point`, using the specified
distance metric function. Results are returned in arbitrary order. 'Best' is determined by
performing a comparison of the elements using < (ie, [`std::cmp::Ordering::is_lt`]). Returns an iterator.

# Examples

```rust
    use fixed::FixedU16;
    use fixed::types::extra::U0;
    use kiddo::fixed::kdtree::KdTree;
    use kiddo::fixed::distance::squared_euclidean;

    type FXD = FixedU16<U0>;

    let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::new();

    tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 1);
    tree.add(&[FXD::from_num(20), FXD::from_num(30), FXD::from_num(60)], 102);

    let mut best_n_within_iter = tree.best_n_within(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], FXD::from_num(10), 1, &squared_euclidean);
    let first = best_n_within_iter.next().unwrap();

    assert_eq!(first, 1);
```"#)
    );
}

#[cfg(test)]
mod tests {
    use crate::fixed::distance::manhattan;
    use crate::fixed::kdtree::{Axis, KdTree};
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;

    type FXD = FixedU16<U14>;

    fn n(num: f32) -> FXD {
        FXD::from_num(num)
    }

    #[test]
    fn can_query_best_n_items_within_radius() {
        let mut tree: KdTree<FXD, u32, 2, 4, u32> = KdTree::new();

        let content_to_add: [([FXD; 2], u32); 16] = [
            ([n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);
        let max_qty = 5;

        let query = [n(0.9f32), n(0.7f32)];
        let radius = n(0.8f32);
        let expected = vec![6, 5, 3, 2, 4];

        let result: Vec<_> = tree
            .best_n_within(&query, radius, max_qty, &manhattan)
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query = [
                n(rng.gen_range(0.0f32..0.9f32)),
                n(rng.gen_range(0.0f32..0.9f32)),
            ];
            println!("{}: {}, {}", _i, query[0].to_string(), query[1].to_string());
            let radius = n(0.1f32);
            let expected = linear_search(&content_to_add, &query, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within(&query, radius, max_qty, &manhattan)
                .collect();

            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_best_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let radius: FXD = n(0.6);
        let max_qty = 5;

        let content_to_add: Vec<([FXD; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand_data_fixed_u16_entry::<U14, u32, 4>())
            .collect();

        let mut tree: KdTree<FXD, u32, 4, 4, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[FXD; 4]> = (0..NUM_QUERIES)
            .map(|_| rand_data_fixed_u16_point::<U14, 4>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within(&query_point, radius, max_qty, &manhattan)
                .collect();

            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
        max_qty: usize,
    ) -> Vec<u32> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let dist = manhattan(query_point, &p);
            if dist <= radius {
                if best_items.len() < max_qty {
                    best_items.push(item);
                } else {
                    if item < *best_items.last().unwrap() {
                        best_items.pop().unwrap();
                        best_items.push(item);
                    }
                }
            }
            best_items.sort_unstable();
        }

        best_items
    }
}
