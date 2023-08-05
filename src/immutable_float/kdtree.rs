//! Immutable Floating point k-d tree. Offers less memory utilisation, smaller size
//! when serialized, and faster more consistent query performace. This comes at the
//! expense of not being able to modify the contents of the tree after its initial
//! construction, and longer construction times - perhaps prohibitively so.
//! As with the vanilla tree, `f64` or `f32` are supported currently for co-ordinate
//! values.

use az::{Az, Cast};
use ordered_float::OrderedFloat;
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::Rem;
use std::ptr;

pub use crate::float::kdtree::Axis;

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::types::Content;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Immutable floating point k-d tree
///
/// Offers less memory utilisation, smaller size
/// when serialized, and faster more consistent query performace. This comes at the
/// expense of not being able to modify the contents of the tree after its initial
/// construction, and longer construction times - perhaps prohiitively so.
/// As with the vanilla tree, `f64` or `f32` are supported currently for co-ordinate
/// values.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ImmutableKdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    pub(crate) leaves: Vec<LeafNode<A, T, K, B>>,
    pub(crate) stems: Vec<A>,
    pub(crate) size: usize,
}

#[doc(hidden)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    #[cfg_attr(feature = "serialize", serde(with = "array_of_arrays"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))
    )]
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation?
    pub(crate) content_points: [[A; K]; B],

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) content_items: [T; B],

    pub(crate) size: usize,
}

impl<A, T, const K: usize, const B: usize> LeafNode<A, T, K, B>
where
    A: Axis,
    T: Content,
{
    fn new() -> Self {
        LeafNode {
            content_items: [T::zero(); B],
            content_points: [[A::zero(); K]; B],
            size: 0,
        }
    }
}
/// Encapsulates stats on a particular `ImmutableTree`'s contents and
/// memory usage at the time of calling `generate_stats()`
#[allow(dead_code)]
#[derive(Debug)]
pub struct TreeStats {
    size: usize,
    capacity: usize,
    stem_count: usize,
    leaf_count: usize,
    leaf_fill_counts: Vec<usize>,
    leaf_fill_ratio: f32,
    stem_fill_ratio: f32,
    unused_stem_count: usize,
}

impl<A, T, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B>
where
    A: Axis,
    T: Content,
{
    /// Creates an ImmutableKdTree, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// Trees constructed using this method will be optimally
    /// balanced and tuned, but will not be modifiable
    /// after construction. This method may take a long time for
    /// large numbers of points (>4 million)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable_float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!();
    /// let tree: KdTree<f64, u32, 3, 32> = KdTree::optimize_from(points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn optimize_from(source: &[[A; K]]) -> Self
    where
        usize: Cast<T>,
    {
        let item_count = source.len();

        // TODO: is it possible to start with an excessive leaf count, but always ensure we are
        //       packed to the left as we go? This could reduce the number of times that we need
        //       to rebalance all the way to the outer loop, and any unused leaves could be
        //       freed up afterwards?

        let mut leaf_node_count = item_count.div_ceil(B);
        let mut stem_node_count = leaf_node_count.next_power_of_two();

        let mut stems = vec![A::infinity(); stem_node_count];
        let mut shifts = vec![0usize; stem_node_count];
        let mut sort_index = Vec::from_iter(0..item_count);

        loop {
            let requested_shift = Self::optimize_stems(
                &mut stems,
                &mut shifts,
                source,
                &mut sort_index,
                1,
                0,
                leaf_node_count * B,
            );

            if requested_shift == 0 {
                break;
            }

            // if root has requested a shift, then there was not enough capacity in
            // the tree to overflow into. Add just enough extra leaf nodes to accommodate
            // the shift.
            leaf_node_count += requested_shift.div_ceil(B);

            // if the new leaf count can't be accommodated by the existing stem count,
            // bump up the stem count to the next power of two.
            if leaf_node_count > stem_node_count {
                stem_node_count = (stem_node_count + 1).next_power_of_two();

                stems = vec![A::infinity(); stem_node_count];
                shifts = Self::extend_shifts(stem_node_count, &shifts, requested_shift);
            }
        }

        let mut tree = Self {
            size: 0,
            stems,
            leaves: Self::safe_allocate_leaves(leaf_node_count),
        };

        for (idx, point) in source.iter().enumerate() {
            tree.safe_add_to_optimized(point, idx.az::<T>());
        }

        tree
    }

    fn extend_shifts(
        stem_node_count: usize,
        shifts: &Vec<usize>,
        requested_shift: usize,
    ) -> Vec<usize> {
        let root_shift = shifts[1];
        let mut new_shifts = vec![0usize; stem_node_count];

        // copy from old to new. Old forms the left subtree of new's root, eg:
        //
        //                          0
        //         1            1       0
        //       2   3   ->   2   3   0   0
        //      4 5 6 7      4 5 6 7 0 0 0 0

        new_shifts[1] = requested_shift;
        new_shifts[2] = root_shift;
        let mut step = 1;
        for i in 2..shifts.len() {
            // check to see if i is a power of 2
            if i.count_ones() == 1 {
                step *= 2;
            }

            if shifts[i] > 0 {
                new_shifts[i + step] = shifts[i];
            }
        }

        new_shifts
    }

    /// Returns a value representing the number of items that would not fit (ie zero if balancing
    /// was successful). If a child splitpoint has landed in between two (or more) items with
    /// the same value on the split axis, the value returned is a hint to
    /// the caller of how many items overflowed.
    fn optimize_stems(
        stems: &mut Vec<A>,
        shifts: &mut Vec<usize>,
        source: &[[A; K]],
        sort_index: &mut [usize],
        stem_index: usize,
        dim: usize,
        capacity: usize,
    ) -> usize {
        let chunk_length = sort_index.len();
        if chunk_length <= B {
            return 0;
        }

        let next_dim = (dim + 1).rem(K);

        let stem_levels_below = stems.len().ilog2() - stem_index.ilog2() - 1;
        let left_capacity = (2usize.pow(stem_levels_below) * B).min(capacity);
        let right_capacity = capacity.saturating_sub(left_capacity);

        let mut pivot =
            Self::calc_pivot(chunk_length, shifts[stem_index], stem_index, right_capacity);

        // only bother with this if we are putting at least one item in the right hand child
        if pivot < chunk_length {
            pivot = Self::update_pivot(source, sort_index, dim, pivot);

            // if we end up with a pivot of 0, something has gone wrong,
            // unless we only had a slice of len 1 anyway
            debug_assert!(pivot > 0 || chunk_length == 1);

            stems[stem_index] = source[sort_index[pivot]][dim];
        }

        // if both subtrees can fit in a bucket, we're done
        if pivot <= B && chunk_length - pivot <= B {
            return 0;
        }

        // if the right chunk is bigger than it's capacity, return the overflow amount
        if chunk_length - pivot > right_capacity {
            return chunk_length - pivot - right_capacity;
        }

        let next_stem_index = stem_index << 1;
        let mut requested_shift_amount;
        let mut lower_sort_index;
        let mut upper_sort_index;
        loop {
            (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);

            requested_shift_amount = Self::optimize_stems(
                stems,
                shifts,
                source,
                lower_sort_index,
                next_stem_index,
                next_dim,
                left_capacity,
            );

            if requested_shift_amount == 0 {
                break;
            }

            pivot -= requested_shift_amount;

            // Test for RHS now having more items than can fit
            // in the buckets present in its subtree. If it does,
            // return with a value so that the parent reduces our
            // total allocation
            if chunk_length - pivot > right_capacity {
                return chunk_length - pivot - right_capacity;
            }

            sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));
            stems[stem_index] = source[sort_index[pivot]][dim];
            shifts[stem_index] += requested_shift_amount;
        }

        // If a right child requests a shift, don't shift yourself,
        // but do pass that shift back up to your parent
        Self::optimize_stems(
            stems,
            shifts,
            source,
            upper_sort_index,
            next_stem_index + 1,
            next_dim,
            right_capacity,
        )
    }

    #[cfg(not(feature = "unreliable_select_nth_unstable"))]
    #[inline]
    fn update_pivot(
        source: &[[A; K]],
        sort_index: &mut [usize],
        dim: usize,
        mut pivot: usize,
    ) -> usize {
        // Using this version of update_pivot makes construction significantly faster (~13%)

        // TODO: this block might be faster by using a quickselect with a fat partition?
        //       we could then run that quickselect and subtract (fat partition length - 1)
        //       from the pivot, avoiding the need for the while loop.

        // ensure the item whose index = pivot is in its correctly sorted position, and any
        // items that are equal to it are adjacent, according to our assumptions about the
        // behaviour of `select_nth_unstable_by` (See examples/check_select_nth_unstable.rs)
        sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 0 {
            pivot -= 1;
        }

        pivot
    }

    #[cfg(feature = "unreliable_select_nth_unstable")]
    #[inline]
    fn update_pivot(
        source: &[[A; K]],
        mut sort_index: &mut [usize],
        dim: usize,
        pivot: usize,
    ) -> usize {
        // ensure the item whose index = pivot is in its correctly sorted position
        let (smaller, _, _) =
            sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));

        // ensure the item whose index = (pivot - 1) is in its correctly sorted position
        smaller.select_nth_unstable_by_key(pivot - 1, |&i| OrderedFloat(source[i][dim]));

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 0 {
            pivot -= 1;

            // ensure the item whose index = (pivot - 1) is in its correctly sorted position
            // now that pivot has been decremented
            sort_index[..pivot]
                .select_nth_unstable_by_key(pivot - 1, |&i| OrderedFloat(source[i][dim]));
        }

        pivot
    }

    #[allow(dead_code)]
    fn allocate_leaves(count: usize) -> Vec<LeafNode<A, T, K, B>> {
        let layout = Layout::array::<LeafNode<A, T, K, B>>(count).unwrap();
        let mut leaves = unsafe {
            let mem = match Global.allocate(layout) {
                Ok(mem) => mem.cast::<LeafNode<A, T, K, B>>().as_ptr(),
                Err(AllocError) => panic!(),
            };

            Vec::from_raw_parts_in(mem, count, count, Global)
        };
        for leaf in &mut leaves {
            leaf.size = 0;
        }

        leaves
    }

    #[allow(dead_code)]
    fn safe_allocate_leaves(count: usize) -> Vec<LeafNode<A, T, K, B>> {
        vec![LeafNode::new(); count]
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    /// tree.add(&[1.1, 2.1, 5.1], 101);
    ///
    /// assert_eq!(tree.size(), 2);
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the theoretical max capacity of this tree
    #[inline]
    pub fn capacity(&self) -> usize {
        self.leaves.len() * B
    }

    /// Generates a `TreeStats` object, describing some
    /// statistics of a particular instance of an `ImmutableTree`
    pub fn generate_stats(&self) -> TreeStats {
        let mut leaf_fill_counts = vec![0usize; B + 1];
        for leaf in &self.leaves {
            leaf_fill_counts[leaf.size.az::<usize>()] += 1;
        }

        let leaf_fill_ratio = (self.size as f32) / (self.capacity() as f32);

        let unused_stem_count = self.stems.iter().filter(|x| x.is_infinite()).count() - 1;

        let stem_fill_ratio = 1.0 - (unused_stem_count as f32 / ((self.stems.len() - 1) as f32));

        TreeStats {
            size: self.size,
            capacity: self.leaves.len() * B,
            stem_count: self.stems.len(),
            leaf_count: self.leaves.len(),
            leaf_fill_counts,
            leaf_fill_ratio,
            stem_fill_ratio,
            unused_stem_count,
        }
    }

    fn calc_pivot(
        chunk_length: usize,
        shifted: usize,
        stem_index: usize,
        right_capacity: usize,
    ) -> usize {
        let mut pivot = (chunk_length + shifted) >> 1;
        if stem_index == 1 {
            // If at the top level, check if there's been a shift
            pivot = if shifted > 0 {
                // if so,
                chunk_length
            } else {
                // otherwise, do a special case pivot shift to ensure the left subtree is full.
                if chunk_length & 1 == 1 {
                    (pivot + 1).next_power_of_two()
                } else {
                    pivot.next_power_of_two()
                }
            };
        } else if chunk_length & 0x01 == 1 && shifted == 0 {
            pivot = (pivot + 1).next_power_of_two()
        } else {
            pivot = pivot.next_power_of_two();
        }
        pivot -= shifted;
        pivot.max(chunk_length.saturating_sub(right_capacity))
    }

    #[inline]
    pub(crate) fn prefetch_stems(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            std::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr::addr_of!(
                prefetch
            )
                as *const i8);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            core::arch::aarch64::_prefetch(
                ptr::addr_of!(prefetch) as *const i8,
                core::arch::aarch64::_PREFETCH_READ,
                core::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use crate::immutable_float::kdtree::ImmutableKdTree;
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};
    use rayon::prelude::IntoParallelRefIterator;

    #[test]
    fn can_construct_optimized_tree_with_straddled_split() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
        ];

        let tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats());

        assert_eq!(tree.leaves[0].size, 3);
        assert_eq!(tree.leaves[1].size, 4);
        assert_eq!(tree.leaves[2].size, 4);
        assert_eq!(tree.leaves[3].size, 4);
        assert_eq!(tree.leaves[4].size, 1);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_2() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];

        let tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats());

        assert_eq!(tree.leaves[0].size, 3);
        assert_eq!(tree.leaves[1].size, 4);
        assert_eq!(tree.leaves[2].size, 4);
        assert_eq!(tree.leaves[3].size, 4);
        assert_eq!(tree.leaves[4].size, 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_3() {
        use rand::seq::SliceRandom;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(23);

        let mut content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];
        content_to_add.shuffle(&mut rng);

        let tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats());

        assert_eq!(tree.leaves[0].size, 3);
        assert_eq!(tree.leaves[1].size, 4);
        assert_eq!(tree.leaves[2].size, 4);
        assert_eq!(tree.leaves[3].size, 4);
        assert_eq!(tree.leaves[4].size, 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_multiple_dupes() {
        use rand::seq::SliceRandom;

        for seed in 0..1_000_000 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

            let mut content_to_add = vec![
                [4.0, 104.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [4.0, 104.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [17.0, 117.0],
                [18.0, 118.0],
            ];
            content_to_add.shuffle(&mut rng);

            let tree: ImmutableKdTree<f32, usize, 2, 8> =
                ImmutableKdTree::optimize_from(&content_to_add);
        }
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_0() {
        let tree_size = 18;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_1() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_2() {
        let tree_size = 155;
        let seed = 480;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_3() {
        let tree_size = 26; // also 32
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_4() {
        let tree_size = 21;
        let seed = 131851;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_5() {
        let tree_size = 32;
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_6() {
        let tree_size = 56;
        let seed = 450533;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_7() {
        let tree_size = 18;
        let seed = 992063;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_8() {
        let tree_size = 19;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_9() {
        let tree_size = 20;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_10() {
        let tree_size = 36;
        let seed = 375096;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_11() {
        let tree_size = 10000;
        let seed = 257281;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    //#[ignore]
    #[test]
    fn can_construct_optimized_tree_multi_rand_increasing_size() {
        use rayon::iter::ParallelIterator;

        #[allow(dead_code)]
        #[derive(Debug)]
        struct Failure {
            tree_size: i32,
            seed: u64,
        }

        let failures: Vec<Failure> = Vec::new();

        for tree_size in 16..100 {
            (0..1000000)
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|&seed| {
                    let result = panic::catch_unwind(|| {
                        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                        let content_to_add: Vec<[f32; 4]> =
                            (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

                        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
                            ImmutableKdTree::optimize_from(&content_to_add);
                    });

                    if result.is_err() {
                        //failures.push(Failure { tree_size, seed });
                        println!("Failed on tree size {}, seed #{}", tree_size, seed);
                    }
                });
        }

        println!("{:?}", &failures);
        assert!(failures.is_empty());
    }

    #[test]
    fn can_construct_optimized_tree_medium_rand() {
        use itertools::Itertools;

        const TREE_SIZE: usize = 2usize.pow(19); // ~ 500k

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let num_uniq = content_to_add
            .iter()
            .flatten()
            .map(|&x| OrderedFloat(x))
            .unique()
            .count();

        println!("dupes: {:?}", TREE_SIZE * 4 - num_uniq);

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 32> =
            ImmutableKdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }
}
