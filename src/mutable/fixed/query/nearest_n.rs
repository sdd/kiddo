use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::mutable::fixed::kdtree::KdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::rkyv_utils::transform;
use crate::traits::{is_stem_index, AxisFixed, Content, DistanceMetricFixed, Index};

use crate::generate_fixed_nearest_n;

impl<A: AxisFixed, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_fixed_nearest_n!(
        (r#"Finds the nearest `qty` elements to `query`, using the specified
distance metric function.

# Examples

```rust
    use fixed::{FixedU16, FixedU32};
    use fixed::types::extra::U0;
    use kiddo::mutable::fixed::kdtree::KdTree;
    use kiddo::distance::fixed::SquaredEuclidean;

    type Fxd = FixedU16<U0>;
    type FxdR = FixedU32<U0>;

    let mut tree: KdTree<Fxd, u32, 3, 32, u32> = KdTree::new();

    tree.add(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], 100);
    tree.add(&[Fxd::from_num(2), Fxd::from_num(3), Fxd::from_num(6)], 101);

    let nearest: Vec<_> = tree.nearest_n::<SquaredEuclidean, FxdR>(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], 1);

    assert_eq!(nearest.len(), 1);
    assert_eq!(nearest[0].distance, FxdR::from_num(0));
    assert_eq!(nearest[0].item, 100);
```"#)
    );
}

#[cfg(test)]
mod tests {
    use crate::distance::fixed::Manhattan;
    use crate::mutable::fixed::kdtree::KdTree;
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
    use crate::traits::{AxisFixed, DistanceMetricFixed};
    use fixed::types::extra::U14;
    use fixed::{FixedU16, FixedU32};
    use rand::Rng;

    type Fxd = FixedU16<U14>;
    type FxdR = FixedU32<U14>;

    fn n(num: f32) -> Fxd {
        Fxd::from_num(num)
    }

    fn nr(num: f32) -> FxdR {
        FxdR::from_num(num)
    }

    #[test]
    fn can_query_nearest_n_items() {
        let mut tree: KdTree<Fxd, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([Fxd; 4], u32); 16] = [
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

        let expected = vec![(nr(0.86), 7), (nr(0.86), 4), (nr(0.86), 5)];

        let result: Vec<_> = tree
            .nearest_n::<Manhattan, FxdR>(&query_point, 3)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                n(rng.random_range(0f32..1f32)),
                n(rng.random_range(0f32..1f32)),
                n(rng.random_range(0f32..1f32)),
                n(rng.random_range(0f32..1f32)),
            ];
            let expected = linear_search(&content_to_add, qty, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<Manhattan, FxdR>(&query_point, qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _): &(FxdR, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    #[test]
    fn can_query_nearest_n_items_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const N: usize = 10;

        let content_to_add: Vec<([Fxd; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand_data_fixed_u16_entry::<U14, u32, 4>())
            .collect();

        let mut tree: KdTree<Fxd, u32, 4, 4, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[Fxd; 4]> = (0..NUM_QUERIES)
            .map(|_| rand_data_fixed_u16_point::<U14, 4>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, N, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<Manhattan, FxdR>(&query_point, N)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _): &(FxdR, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    fn linear_search<A: AxisFixed, R: AxisFixed, const K: usize>(
        content: &[([A; K], u32)],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(R, u32)> {
        let mut results = vec![];

        for &(p, item) in content {
            let dist: R = Manhattan::dist(query_point, &p);
            if results.len() < qty {
                results.push((dist, item));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, item);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
    }
}
