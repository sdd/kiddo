use az::{Az, Cast};
use std::ops::Rem;

use crate::fixed::kdtree::{Axis, KdTree};
use crate::types::{Content, Index};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    pub fn within_unsorted<F>(&self, query: &[A; K], radius: A, distance_fn: &F) -> Vec<(A, T)>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut matching_items = Vec::new();

        unsafe {
            self.within_unsorted_recurse(
                query,
                radius,
                distance_fn,
                self.root_index,
                0,
                &mut matching_items,
            );
        }

        matching_items
    }

    unsafe fn within_unsorted_recurse<F>(
        &self,
        query: &[A; K],
        radius: A,
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        matching_items: &mut Vec<(A, T)>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let child_node_indices = if *query.get_unchecked(split_dim) < node.split_val {
                [node.left, node.right]
            } else {
                [node.right, node.left]
            };
            let next_split_dim = (split_dim + 1).rem(K);

            for node_idx in child_node_indices {
                let child_node_dist = self.child_dist_to_bounds(query, node_idx, distance_fn);
                if child_node_dist <= radius {
                    self.within_unsorted_recurse(
                        query,
                        radius,
                        distance_fn,
                        node_idx,
                        next_split_dim,
                        matching_items,
                    );
                }
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());
            // println!("Leaf node: {:?}", (curr_node_idx - LEAF_OFFSET) as usize);

            leaf_node
                .content_points
                .iter()
                .enumerate()
                .take(leaf_node.size.az::<usize>())
                .for_each(|(idx, entry)| {
                    let distance = distance_fn(query, entry);

                    if distance < radius {
                        matching_items.push((
                            distance,
                            *leaf_node.content_items.get_unchecked(idx.az::<usize>()),
                        ));
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed::distance::manhattan;
    use crate::fixed::kdtree::{Axis, KdTree};
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;
    use std::cmp::Ordering;

    type FXD = FixedU16<U14>;

    fn n(num: f32) -> FXD {
        FXD::from_num(num)
    }

    #[test]
    fn can_query_items_within_radius() {
        let mut tree: KdTree<FXD, u32, 4, 5, u32> = KdTree::new();

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

        let radius = n(0.2);
        let expected = linear_search(&content_to_add, &query_point, radius);

        let result = tree.within_unsorted(&query_point, radius, &manhattan);
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

            let mut result = tree.within_unsorted(&query_point, radius, &manhattan);
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
            let dist = manhattan(query_point, &p);
            if dist < radius {
                matching_items.push((dist, item));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut Vec<(A, u32)>) {
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
