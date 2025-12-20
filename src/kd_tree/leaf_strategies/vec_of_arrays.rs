use crate::kd_tree::leaf_view::LeafView;
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, Mutable, MutableLeafStrategy};
use crate::StemStrategy;
use aligned_vec::AVec;

/// A leaf storage strategy using vectors of fixed-size arrays.
///
/// Stores each leaf as a fixed-size array for better cache locality.
pub struct VecOfArrays<A, T, const K: usize, const B: usize> {
    leaves: Vec<LeafNode<A, T, K, B>>,
    size: usize,
}

/// A single leaf node storing up to B points.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A, T, const K: usize, const B: usize> {
    /// Point coordinates organized by dimension
    pub content_points: [[A; B]; K],
    /// Items associated with each point
    pub content_items: [T; B],
    /// Number of points currently stored
    pub size: usize,
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics,
    SS: StemStrategy,
{
    type Num = AX;
    type Mutability = Mutable;

    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaves: Vec::with_capacity(capacity / B + 1),
            size: 0,
        }
    }

    fn new_with_empty_leaf() -> Self {
        let mut result = Self {
            leaves: Vec::new(),
            size: 0,
        };

        let leaf = LeafNode {
            content_points: [[AX::zero(); B]; K],
            content_items: [T::default(); B],
            size: 0,
        };

        result.leaves.push(leaf);

        result
    }

    fn bulk_build_from_slice(
        &mut self,
        _source: &[[Self::Num; K]],
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: SS,
    ) -> i32 {
        todo!()
    }

    fn finalize(
        &mut self,
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: &mut SS,
        _max_stem_level: i32,
    ) {
        todo!()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        todo!()
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let leaf = &self.leaves[leaf_idx];

        let points: [&[AX]; K] =
            array_init::array_init(|i| &leaf.content_points[i].as_slice()[..leaf.size]);
        let leaf_items_view = &leaf.content_items[..leaf.size];

        LeafView::new(points, leaf_items_view)
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let leaf_len = leaf_items.len();
        debug_assert!(leaf_len <= B);

        // Sanity: all dims should have the same length
        debug_assert!(leaf_points.iter().all(|p| p.len() == leaf_len));

        // Initialize fixed-size storage with defaults
        let mut content_points: [[AX; B]; K] = array_init::array_init(|_| [AX::zero(); B]);
        let mut content_items: [T; B] = [T::default(); B];

        // Copy the actual data into the first `leaf_len` slots
        for i in 0..leaf_len {
            for dim in 0..K {
                content_points[dim][i] = leaf_points[dim][i];
            }
            content_items[i] = leaf_items[i];
        }

        self.leaves.push(LeafNode {
            content_points,
            content_items,
            size: leaf_len,
        });
        self.size += leaf_len;
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified,
    T: Copy + PartialEq,
{
    fn should_remove(
        leaf: &LeafNode<AX, T, K, B>,
        point: &[AX; K],
        item: T,
        leaf_idx: usize,
    ) -> bool {
        for dim in 0..K {
            if leaf.content_points[dim][leaf_idx] != point[dim] {
                return false;
            }
        }

        if leaf.content_items[leaf_idx] != item {
            return false;
        }

        true
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics + PartialEq,
{
    fn get_split_value(&self, leaf_idx: usize, pivot_idx: usize, split_dim: usize) -> AX {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked(leaf_idx) };
        debug_assert!(pivot_idx < leaf.size, "pivot_idx out of bounds");

        leaf.content_points[split_dim][pivot_idx]
    }

    fn copy_split_data_to_new_leaf(
        &mut self,
        old_leaf: &LeafNode<AX, T, K, B>,
        pivot_idx: usize,
    ) -> usize {
        debug_assert!(pivot_idx < B, "pivot_idx out of bounds");

        // Create a new empty leaf
        let mut new_leaf = LeafNode {
            content_points: [[AX::zero(); B]; K],
            content_items: [T::default(); B],
            size: 0,
        };

        // Copy the right half to the new leaf
        let num_items = B - pivot_idx;
        unsafe {
            // Copy each dimension's data
            for dim in 0..K {
                new_leaf
                    .content_points
                    .get_unchecked_mut(dim)
                    .get_unchecked_mut(..num_items)
                    .copy_from_slice(
                        old_leaf
                            .content_points
                            .get_unchecked(dim)
                            .get_unchecked(pivot_idx..B),
                    );
            }
            // Copy items
            new_leaf
                .content_items
                .get_unchecked_mut(..num_items)
                .copy_from_slice(old_leaf.content_items.get_unchecked(pivot_idx..B));
        }

        // set new leaf size
        new_leaf.size = num_items;

        // Add the new leaf
        self.leaves.push(new_leaf);

        // return the new leaf index
        self.leaves.len() - 1
    }
}

impl<AX, T, SS, const K: usize, const B: usize> MutableLeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics + PartialEq,
    SS: StemStrategy,
{
    fn add_to_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let idx = leaf.size;
        debug_assert!(leaf.size < B, "leaf is full (max capacity reached)");

        for dim in 0..K {
            leaf.content_points[dim][idx] = unsafe { *point.get_unchecked(dim) };
        }
        leaf.content_items[idx] = item;

        leaf.size += 1;
    }

    fn is_leaf_full(&self, leaf_idx: usize) -> bool {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");

        let leaf = unsafe { self.leaves.get_unchecked(leaf_idx) };

        leaf.size >= B
    }

    fn remove_from_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let mut new_idx = 0;
        for curr_idx in 0..leaf.size {
            // skip items that need removal
            if Self::should_remove(leaf, point, item, curr_idx) {
                continue;
            }

            // if no items have yet needed removal, no action yet needs to be taken
            if new_idx == curr_idx {
                new_idx += 1;
                continue;
            }

            // curr item needs to be kept but needs copying down to lower position in leaf
            for dim in 0..K {
                leaf.content_points[dim][new_idx] = leaf.content_points[dim][curr_idx];
            }
            leaf.content_items[new_idx] = leaf.content_items[curr_idx];
            new_idx += 1;
        }
        self.size -= leaf.size - new_idx;
        leaf.size = new_idx;
    }

    /// Splits this leaf by:
    /// * sorting it along split_dim
    /// * finding the midpoint
    /// * creating a new empty leaf
    /// * moving items from the midpoint onwards to the new leaf
    ///
    /// Returns: A tuple of (split_value, new_leaf_idx)
    fn split_leaf(&mut self, leaf_idx: usize, split_dim: usize) -> (AX, usize) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        unsafe {
            let orig = self.leaves.get_unchecked_mut(leaf_idx);

            let mut pivot_idx: usize = orig.size / 2;

            // Create a mirror array to track the permutation
            let mut mirror: [usize; B] = array_init::array_init(|i| i);

            // Sort the split dimension and track the permutation in mirror
            mirror_select_nth_unstable_by(
                &mut orig.content_points[split_dim][..orig.size],
                &mut mirror[..orig.size],
                pivot_idx,
                |a, b| a.partial_cmp(b).expect("Leaf node sort failed."),
            );

            // Apply the permutation to all other dimensions and items
            for dim in 0..K {
                if dim != split_dim {
                    let temp = orig.content_points[dim];
                    for i in 0..orig.size {
                        orig.content_points[dim][i] = temp[mirror[i]];
                    }
                }
            }
            let temp_items = orig.content_items;
            for i in 0..orig.size {
                orig.content_items[i] = temp_items[mirror[i]];
            }

            let mut split_val = *orig
                .content_points
                .get_unchecked(split_dim)
                .get_unchecked(pivot_idx);

            // At this point we hava a candidate position at which to split the leaf, that
            // being the midpoint. This may not be a valid location, however. Take this 1D case
            // as an example. We've sorted our leaf and it looks like this:
            //
            // [1, 2, 2, 4, 5]
            //
            // If we choose the actual midpoint and move it, and everything to the right of it,
            // to a new leaf, and we choose 2 as the new stem value, what about the 2 that was to the
            // left of it? It is now in the wrong bucket. So, we need to adjust our split point by
            // moving it one place to the left.
            //
            // What about in this situation though?
            //
            // [2, 2, 2, 4, 5]
            //
            // If we follow the same algorithm, we end up with a split that has zero items in the
            // left bucket and 5 on the right. This is no good! If we're splitting so that we can add
            // a 4, for example, once the split completes, we still can't add it!
            //
            // So, if we've tried to adjust the split point leftwards, and got all the way to the
            // start of the bucket, we reset back to the midpoint and start moving rightwards.
            // In the above case this ends up with us splitting into [2, 2, 2] and [4, 5], with
            // the pivot value being 4.
            //
            // But what if we had a bucket that looked like this?
            //
            // [2, 2, 2, 2, 2]
            //
            // Well then, to put it bluntly, we're fucked. This VecOfArrays leaf strategy has
            // fixed-size buckets by design, to avoid the extra layer of indirection that would result
            // from using a Vec for the bucket. There is no way to store another 2 in the tree - you need
            // to either increase the bucket size or switch to a different, more permissive leaf strategy.

            // if the pivot point choice results in some items whose position on the split
            // dimension is the same as the split value being on the wrong side of the split:
            if *orig
                .content_points
                .get_unchecked(split_dim)
                .get_unchecked(pivot_idx - 1)
                == split_val
            {
                let orig_pivot_idx = pivot_idx;

                // Ensure that if pivot index would result in items that share the same co-ordinate
                // on the splitting dimension would end up on different sides of the split, that we
                // move the pivot to prevent this. We first try moving down, since that's the only
                // part of the bucket that was sorted by mirror_select_nth_unstable_by
                while pivot_idx > 0
                    && *orig
                        .content_points
                        .get_unchecked(split_dim)
                        .get_unchecked(pivot_idx - 1)
                        == split_val
                {
                    pivot_idx -= 1;
                }

                // If the attempt to move the pivot point above would have resulted in the pivot
                // point moving to the start of the bucket, search forwards from the original
                // pivot point instead. We need to first ensure the upper half of the bucket
                // is sorted
                if pivot_idx == 0 {
                    // Re-sort to ensure the entire array is sorted
                    mirror_select_nth_unstable_by(
                        &mut orig.content_points[split_dim][..orig.size],
                        &mut mirror[..orig.size],
                        orig.size - 1,
                        |a, b| a.partial_cmp(b).expect("Leaf node sort failed."),
                    );

                    // Re-apply the permutation to all other dimensions and items
                    for dim in 0..K {
                        if dim != split_dim {
                            let temp = orig.content_points[dim];
                            for i in 0..orig.size {
                                orig.content_points[dim][i] = temp[mirror[i]];
                            }
                        }
                    }
                    let temp_items = orig.content_items;
                    for i in 0..orig.size {
                        orig.content_items[i] = temp_items[mirror[i]];
                    }

                    pivot_idx = orig_pivot_idx;
                    while *orig
                        .content_points
                        .get_unchecked(split_dim)
                        .get_unchecked(pivot_idx)
                        == split_val
                    {
                        pivot_idx += 1;

                        if pivot_idx == B {
                            // TODO: should no longer panic here. Changing addition to be fallible
                            // is quite a wide-ranging change though
                            panic!("Too many items with the same position on one axis. Bucket size must be increased to at least 1 more than the number of items with the same position on one axis.");
                        }
                    }
                }

                split_val = *orig
                    .content_points
                    .get_unchecked(split_dim)
                    .get_unchecked(pivot_idx);
            }

            // At this point, we have a valid index at which we can split the bucket.
            // We need to copy the old leaf data before we can borrow self again
            let old_leaf_copy = orig.clone();
            orig.size = pivot_idx;

            let new_leaf_idx = self.copy_split_data_to_new_leaf(&old_leaf_copy, pivot_idx);

            (split_val, new_leaf_idx)
        }
    }
}

#[cfg(test)]
mod test {
    use fixed::{types::extra::U8, FixedU16};
    use rand::Rng;

    use crate::kd_tree::leaf_strategies::vec_of_arrays::VecOfArrays;
    use crate::traits_unified_2::{LeafStrategy, SquaredEuclidean};
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn create_single_leaf_vec_of_arrays_float_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view = <VecOfArrays<f32, u32, 3, 32> as LeafStrategy<
            f32,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::leaf_view(&tree.leaves, 0);

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_vec_of_arrays_fixed_point_kd_tree() {
        let points: Vec<[FixedU16<U8>; 3]> = vec![[1.into(), 2.into(), 3.into()]];
        let tree: kd_tree::KdTree<
            FixedU16<U8>,
            u32,
            Eytzinger<3>,
            VecOfArrays<FixedU16<U8>, u32, 3, 32>,
            3,
            32,
        > = kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view = <VecOfArrays<FixedU16<U8>, u32, 3, 32> as LeafStrategy<
            FixedU16<U8>,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::leaf_view(&tree.leaves, 0);
        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_vec_of_arrays_float_no_items_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];

        let tree: kd_tree::KdTree<f32, (), Eytzinger<3>, VecOfArrays<f32, (), 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice_no_items(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <VecOfArrays<f32, (), 3, 32> as LeafStrategy<f32, (), Eytzinger<3>, 3, 32>>::leaf_view(
                &tree.leaves,
                0,
            );

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![()]);
    }

    #[test]
    fn create_multiple_leaf_vec_of_arrays_float_kd_tree() {
        // create 2^16 random 3d points in the unit cube
        let mut rng = rand::rng();
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        // perform a nearest_one query
        let query_point = [0.5, 0.5, 0.5];

        let _nearest = tree.nearest_one::<SquaredEuclidean<f32>>(&query_point);
    }
}
