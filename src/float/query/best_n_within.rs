use crate::float::kdtree::{Axis, KdTree, LeafNode};

use crate::types::{is_stem_index, Content, Index};
use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::generate_best_n_within;

macro_rules! generate_float_best_n_within {
    ($leafnode:ident, $doctest_build_tree:tt) => {
        generate_best_n_within!(
            $leafnode,
            (
                "Finds the \"best\" `n` elements within `dist` of `query`.

Results are returned in arbitrary order. 'Best' is determined by
performing a comparison of the elements using < (ie, std::ord::lt).
Returns an iterator.

# Examples

```rust
    use kiddo::float::kdtree::KdTree;
    use kiddo::distance::squared_euclidean;

    ",
                $doctest_build_tree,
                "

    let mut best_n_within = tree.best_n_within(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean);
    let first = best_n_within.next().unwrap();

    assert_eq!(first, 100);
```"
            )
        );
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_best_n_within!(
        LeafNode,
        "let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::{ArchivedKdTree, ArchivedLeafNode};
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>,
    > ArchivedKdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_best_n_within!(
        ArchivedLeafNode,
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/test-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<KdTree<f64, u32, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::squared_euclidean;
    use crate::float::kdtree::KdTree;
    use rand::Rng;

    type AX = f64;

    #[test]
    fn can_query_best_n_items_within_radius() {
        let mut tree: KdTree<AX, i32, 2, 4, u32> = KdTree::new();

        let content_to_add = [
            ([9f64, 0f64], 9),
            ([4f64, 500f64], 4),
            ([12f64, -300f64], 12),
            ([7f64, 200f64], 7),
            ([13f64, -400f64], 13),
            ([6f64, 300f64], 6),
            ([2f64, 700f64], 2),
            ([14f64, -500f64], 14),
            ([3f64, 600f64], 3),
            ([10f64, -100f64], 10),
            ([16f64, -700f64], 16),
            ([1f64, 800f64], 1),
            ([15f64, -600f64], 15),
            ([5f64, 400f64], 5),
            ([8f64, 100f64], 8),
            ([11f64, -200f64], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }
        assert_eq!(tree.size(), 16);

        let query = [9f64, 0f64];
        let radius = 20000f64;
        let max_qty = 3;
        let expected = vec![10, 9, 8];

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

        let content_to_add: Vec<([AX; 2], i32)> = (0..TREE_SIZE)
            .map(|_| rand::random::<([AX; 2], i32)>())
            .collect();

        let mut tree: KdTree<AX, i32, 2, 32, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as i32);

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
        content: &[([f64; 2], i32)],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<i32> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let dist = squared_euclidean(query, &p);
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
        best_items.reverse();

        best_items
    }
}
