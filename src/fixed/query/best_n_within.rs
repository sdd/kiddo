use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::best_neighbour::BestNeighbour;
use crate::fixed::kdtree::{Axis, KdTree, LeafNode};
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};

use crate::generate_best_n_within;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_best_n_within!(
        LeafNode,
        (r#"Queries the tree to find the best `n` elements within `dist` of `point`, using the specified
distance metric.

Returns an iterator.
Results are returned in arbitrary order. 'Best' is determined by
performing a comparison of the elements using < (ie, [`std::cmp::Ordering::is_lt`]).

# Examples

```rust
    use fixed::FixedU16;
    use fixed::types::extra::U0;
    use kiddo::best_neighbour::BestNeighbour;
    use kiddo::fixed::kdtree::KdTree;
    use kiddo::fixed::distance::SquaredEuclidean;

    type Fxd = FixedU16<U0>;

    let mut tree: KdTree<Fxd, u32, 3, 32, u32> = KdTree::new();

    tree.add(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], 100);
    tree.add(&[Fxd::from_num(2), Fxd::from_num(3), Fxd::from_num(6)], 1);
    tree.add(&[Fxd::from_num(20), Fxd::from_num(30), Fxd::from_num(60)], 102);

    let mut best_n_within_iter = tree.best_n_within::<SquaredEuclidean>(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], Fxd::from_num(10), 1);
    let first = best_n_within_iter.next().unwrap();

    assert_eq!(first, BestNeighbour { distance: Fxd::from_num(3), item: 1 });
```"#)
    );
}

#[cfg(test)]
mod tests {
    use crate::best_neighbour::BestNeighbour;
    use crate::fixed::distance::Manhattan;
    use crate::fixed::kdtree::{Axis, KdTree};
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
    use crate::traits::DistanceMetric;
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;

    type Fxd = FixedU16<U14>;

    fn n(num: f32) -> Fxd {
        Fxd::from_num(num)
    }

    #[test]
    fn can_query_best_n_items_within_radius() {
        let mut tree: KdTree<Fxd, u32, 2, 4, u32> = KdTree::new();

        let content_to_add: [([Fxd; 2], u32); 16] = [
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
        let expected = vec![
            BestNeighbour {
                distance: n(0.7001f32),
                item: 6,
            },
            BestNeighbour {
                distance: n(0.7f32),
                item: 5,
            },
            BestNeighbour {
                distance: n(0.7001f32),
                item: 3,
            },
            BestNeighbour {
                distance: n(0.7f32),
                item: 2,
            },
            BestNeighbour {
                distance: n(0.7f32),
                item: 4,
            },
        ];

        let result: Vec<_> = tree
            .best_n_within::<Manhattan>(&query, radius, max_qty)
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query = [
                n(rng.random_range(0.0f32..0.9f32)),
                n(rng.random_range(0.0f32..0.9f32)),
            ];
            let radius = n(0.1f32);
            let expected = linear_search(&content_to_add, &query, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within::<Manhattan>(&query, radius, max_qty)
                .collect();

            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_best_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let radius: Fxd = n(0.6);
        let max_qty = 5;

        let content_to_add: Vec<([Fxd; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand_data_fixed_u16_entry::<U14, u32, 4>())
            .collect();

        let mut tree: KdTree<Fxd, u32, 4, 4, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[Fxd; 4]> = (0..NUM_QUERIES)
            .map(|_| rand_data_fixed_u16_point::<U14, 4>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within::<Manhattan>(&query_point, radius, max_qty)
                .collect();

            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query: &[A; K],
        radius: A,
        max_qty: usize,
    ) -> Vec<BestNeighbour<A, u32>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let distance = Manhattan::dist(query, &p);
            if distance <= radius {
                if best_items.len() < max_qty {
                    best_items.push(BestNeighbour { distance, item });
                } else if item < best_items.last().unwrap().item {
                    best_items.pop().unwrap();
                    best_items.push(BestNeighbour { distance, item });
                }
            }
            best_items.sort_unstable();
        }

        best_items
    }
}
