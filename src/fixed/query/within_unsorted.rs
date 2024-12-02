use az::{Az, Cast};
use std::ops::Rem;

use crate::fixed::kdtree::{Axis, KdTree};
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};

use crate::generate_within_unsorted;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_within_unsorted!(
        (r#"Finds all elements within `dist` of `query`, using the specified
distance metric function.

Results are returned in arbitrary order. Faster than `within`.

# Examples

```rust
    use fixed::FixedU16;
    use fixed::types::extra::U0;
    use kiddo::fixed::kdtree::KdTree;
    use kiddo::fixed::distance::SquaredEuclidean;

    type Fxd = FixedU16<U0>;


    let mut tree: KdTree<Fxd, u32, 3, 32, u32> = KdTree::new();

    tree.add(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], 100);
    tree.add(&[Fxd::from_num(2), Fxd::from_num(3), Fxd::from_num(6)], 101);
    tree.add(&[Fxd::from_num(20), Fxd::from_num(30), Fxd::from_num(60)], 102);

    let within = tree.within::<SquaredEuclidean>(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], Fxd::from_num(10));

    assert_eq!(within.len(), 2);
```"#)
    );
}

#[cfg(test)]
mod tests {
    use crate::fixed::distance::Manhattan;
    use crate::fixed::kdtree::{Axis, KdTree};
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
    use crate::traits::DistanceMetric;
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;
    use std::cmp::Ordering;

    type Fxd = FixedU16<U14>;

    fn n(num: f32) -> Fxd {
        Fxd::from_num(num)
    }

    #[test]
    fn can_query_items_within_radius() {
        let mut tree: KdTree<Fxd, u32, 4, 5, u32> = KdTree::new();

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

        let radius = n(0.2);
        let expected = linear_search(&content_to_add, &query_point, radius);

        let result: Vec<_> = tree
            .within_unsorted::<Manhattan>(&query_point, radius)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
            ];
            let radius = n(2.0);
            let expected = linear_search(&content_to_add, &query_point, radius);

            let mut result: Vec<_> = tree
                .within_unsorted::<Manhattan>(&query_point, radius)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let radius: Fxd = n(0.2);

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
            let expected = linear_search(&content_to_add, &query_point, radius);

            let mut result: Vec<_> = tree
                .within_unsorted::<Manhattan>(&query_point, radius)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for &(p, item) in content {
            let dist = Manhattan::dist(query_point, &p);
            if dist < radius {
                matching_items.push((dist, item));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut [(A, u32)]) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}
