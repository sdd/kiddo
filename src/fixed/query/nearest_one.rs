use az::{Az, Cast};
use std::ops::Rem;

use crate::fixed::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{is_stem_index, Content, Index};

use crate::generate_nearest_one;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_nearest_one!(
        LeafNode,
        (r#"Queries the tree to find the nearest element to `query`, using the specified
distance metric function.

Faster than querying for nearest_n(point, 1, ...) due
to not needing to allocate memory or maintain sorted results.

# Examples

```rust
    use fixed::FixedU16;
    use fixed::types::extra::U0;
    use kiddo::fixed::kdtree::KdTree;
    use kiddo::fixed::distance::squared_euclidean;

    type FXD = FixedU16<U0>;

    let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::new();

    tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 101);

    let nearest = tree.nearest_one(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], &squared_euclidean);

    assert_eq!(nearest.0, FXD::from_num(0));
    assert_eq!(nearest.1, 100);
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
    fn can_query_nearest_one_item() {
        let mut tree: KdTree<FXD, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([FXD; 4], u32); 16] = [
            ([n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32), n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32), n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32), n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32), n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32), n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32), n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32), n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32), n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32), n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32), n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32), n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32), n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32), n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32), n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32), n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [n(0.78f32), n(0.55f32), n(0.78f32), n(0.55f32)];
        let expected = (n(0.86), 7);

        let result = tree.nearest_one(&query_point, &manhattan);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            assert_eq!(result.0, expected.0);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

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
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            assert_eq!(result.0, expected.0);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
    ) -> (A, u32) {
        let mut best_dist: A = A::max_value();
        let mut best_item: u32 = u32::MAX;

        for &(p, item) in content {
            let dist = manhattan(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
