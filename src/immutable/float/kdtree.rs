//! Immutable Floating point k-d tree. Offers less memory utilisation, smaller size
//! when serialized, and faster more consistent query performance. This comes at the
//! expense of not being able to modify the contents of the tree after its initial
//! construction, and longer construction times - perhaps prohibitively so.
//! As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
//! values.

use az::{Az, Cast};
use ordered_float::OrderedFloat;
use std::cmp::PartialEq;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Rem;
#[cfg(feature = "tracing")]
use tracing::{event, span, Level};

pub use crate::float::kdtree::Axis;
use crate::float_leaf_simd::leaf_node::{BestFromDists, LeafNode};
use crate::iter::{IterableTreeData, TreeIter};
use crate::types::Content;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Immutable floating point k-d tree
///
/// Offers less memory utilisation, smaller size
/// when serialized, and faster more consistent query performance. This comes at the
/// expense of not being able to modify the contents of the tree after its initial
/// construction, and longer construction times.
/// As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
/// values.
///
/// A convenient type alias exists for ImmutableKdTree with some sensible defaults set: [`kiddo::ImmutableKdTree`](`crate::ImmutableKdTree`).
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

impl<A: Axis, T: Content, const K: usize, const B: usize> From<&[[A; K]]>
    for ImmutableKdTree<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = (&*points).into();
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    fn from(slice: &[[A; K]]) -> Self {
        ImmutableKdTree::new_from_slice(slice)
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize> IterableTreeData<A, T, K>
    for ImmutableKdTree<A, T, K, B>
{
    fn get_leaf_data(&self, idx: usize) -> Option<VecDeque<(T, [A; K])>> {
        let leaf = self.leaves.get(idx)?;
        let max = leaf.size;
        let mut pts = VecDeque::with_capacity(max);
        for (pt_idx, content) in leaf.content_items[..max].iter().cloned().enumerate() {
            let mut arr = [A::default(); K];
            for (elem_idx, elem) in arr.iter_mut().enumerate() {
                *elem = leaf.content_points[pt_idx][elem_idx];
            }
            pts.push_back((content, arr));
        }
        Some(pts)
    }
}

impl<A, T, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn new_from_slice(source: &[[A; K]]) -> Self
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

        //let mut top_level_shift_change_count = 0;
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
            //top_level_shift_change_count += 1;

            // if root has requested a shift, then there was not enough capacity in
            // the tree to overflow into. Add just enough extra leaf nodes to accommodate
            // the shift.
            leaf_node_count += requested_shift.div_ceil(B);
            #[cfg(feature = "tracing")]
            event!(
                Level::TRACE,
                requested_shift,
                leaf_node_count,
                "top-level shift"
            );

            // if the new leaf count can't be accommodated by the existing stem count,
            // bump up the stem count to the next power of two.
            if leaf_node_count > stem_node_count {
                stem_node_count = (stem_node_count + 1).next_power_of_two();
                #[cfg(feature = "tracing")]
                event!(Level::TRACE, stem_node_count, "extending stems");

                stems = vec![A::infinity(); stem_node_count];
                shifts = Self::extend_shifts(stem_node_count, &shifts, requested_shift);
            }
        }

        // // generate construction stats
        // let max_shift = shifts.iter().max().unwrap();
        // let tot_shift: usize = shifts.iter().sum();
        //
        // println!(
        //     "top_level_shift_change_count: {:?}",
        //     top_level_shift_change_count
        // );
        // println!("max_shift: {:?}", max_shift);
        // println!("tot_shift: {:?}", tot_shift);
        //
        // for (idx, &shift) in shifts.iter().enumerate() {
        //     if shift != 0 {
        //         println!("Index {}: shift {}", &idx, shift);
        //     }
        // }

        let mut tree = Self {
            size: 0,
            stems,
            leaves: Self::allocate_leaves(leaf_node_count),
        };

        for (idx, point) in source.iter().enumerate() {
            tree.safe_add_to_optimized(point, idx.az::<T>());
        }

        tree
    }

    fn extend_shifts(
        stem_node_count: usize,
        shifts: &[usize],
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
        #[cfg(feature = "tracing")]
        let span = span!(Level::TRACE, "opt", idx = stem_index);
        #[cfg(feature = "tracing")]
        let _enter = span.enter();
        let chunk_length = sort_index.len();
        if chunk_length <= B {
            return 0;
        }

        assert!(chunk_length <= capacity);

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
            #[cfg(feature = "tracing")]
            event!(
                Level::TRACE,
                val = chunk_length - pivot - right_capacity,
                "RHS Overflow A"
            );
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

            // exit the loop if the LHS balanced
            if requested_shift_amount == 0 {
                break;
            }

            #[cfg(feature = "tracing")]
            event!(Level::TRACE, req = requested_shift_amount, "LHS shift");

            pivot -= requested_shift_amount;
            pivot = Self::update_pivot(source, sort_index, dim, pivot);

            // Test for RHS now having more items than can fit
            // in the buckets present in its subtree. If it does,
            // return with a value so that the parent reduces our
            // total allocation
            if chunk_length - pivot > right_capacity {
                #[cfg(feature = "tracing")]
                event!(Level::TRACE, val = requested_shift_amount, "shift A");
                shifts[stem_index] += requested_shift_amount;

                #[cfg(feature = "tracing")]
                event!(
                    Level::TRACE,
                    val = chunk_length - pivot - right_capacity,
                    "RHS Overflow B"
                );
                return chunk_length - pivot - right_capacity;
            }

            sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));
            stems[stem_index] = source[sort_index[pivot]][dim];

            #[cfg(feature = "tracing")]
            event!(
                Level::TRACE,
                idx = stem_index,
                d = requested_shift_amount,
                "shift B"
            );
            shifts[stem_index] += requested_shift_amount;
        }

        // If a right child requests a shift, don't shift yourself,
        // but do pass that shift back up to your parent
        let res = Self::optimize_stems(
            stems,
            shifts,
            source,
            upper_sort_index,
            next_stem_index + 1,
            next_dim,
            right_capacity,
        );
        #[cfg(feature = "tracing")]
        if res != 0 {
            event!(Level::TRACE, val = res, "RHS shift");
        }

        res
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
            #[cfg(feature = "tracing")]
            event!(
                Level::INFO,
                pivot,
                chunk_len = sort_index.len(),
                pivotN1_val = ?source[sort_index[pivot - 1]][dim],
                pivot0_val = ?source[sort_index[pivot]][dim],
                pivot1_val = ?source[sort_index[pivot + 1]][dim],
                "pivot shifted"
            );
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

    #[cfg(feature = "global_allocate")]
    fn allocate_leaves(count: usize) -> Vec<LeafNode<A, T, K, B>> {
        use std::alloc::{AllocError, Allocator, Global, Layout};

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

    #[cfg(not(feature = "global_allocate"))]
    fn allocate_leaves(count: usize) -> Vec<LeafNode<A, T, K, B>> {
        vec![LeafNode::new(); count]
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
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
            pivot = if chunk_length & 1 == 1 {
                #[cfg(feature = "tracing")]
                event!(Level::DEBUG, "calc_pivot: unusual route");
                (pivot + 1).next_power_of_two()
            } else {
                //event!(Level::TRACE, "cp C");
                pivot.next_power_of_two()
            };
        } else if chunk_length & 0x01 == 1 && shifted == 0 {
            //#[cfg(feature = "tracing")]
            //event!(Level::TRACE, "cp D");
            pivot = (pivot + 1).next_power_of_two()
        } else {
            pivot = pivot.next_power_of_two();
        }
        pivot -= shifted;
        pivot = pivot.max(chunk_length.saturating_sub(right_capacity));
        //#[cfg(feature = "tracing")]
        //event!(Level::TRACE, pivot, "pivot");
        pivot
        // pivot - shifted
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline]
    pub(crate) fn prefetch_stems(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            std::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                std::ptr::addr_of!(prefetch) as *const i8,
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            core::arch::aarch64::_prefetch(
                std::ptr::addr_of!(prefetch) as *const i8,
                core::arch::aarch64::_PREFETCH_READ,
                core::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }

    /// Iterate over all `(index, point)` tuples in arbitrary order.
    ///

    /// ```
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
    ///
    /// let mut pairs: Vec<_> = tree.iter().collect()
    /// assert_eq!(pairs.pop(), (0, [1.0, 2.0, 3.0]));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (T, [A; K])> + '_ {
        TreeIter::new(self)
    }
}

#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > ArchivedImmutableKdTree<A, T, K, B>
{
    /// Returns the current number of elements stored in the tree
    #[inline]
    pub fn size(&self) -> usize {
        self.size as usize
    }
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline]
    pub(crate) fn prefetch_stems(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            std::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                std::ptr::addr_of!(prefetch) as *const i8,
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            core::arch::aarch64::_prefetch(
                std::ptr::addr_of!(prefetch) as *const i8,
                core::arch::aarch64::_PREFETCH_READ,
                core::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, panic};

    use crate::immutable::float::kdtree::ImmutableKdTree;
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
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats());

        assert_eq!(tree.leaves[0].size, 3);
        assert_eq!(tree.leaves[1].size, 4);
        assert_eq!(tree.leaves[2].size, 4);
        assert_eq!(tree.leaves[3].size, 1);
        assert_eq!(tree.leaves[4].size, 4);
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
            ImmutableKdTree::new_from_slice(&content_to_add);

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
            ImmutableKdTree::new_from_slice(&content_to_add);

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

        for seed in 0..1_000 {
            //_000 {
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

            let _tree: ImmutableKdTree<f32, usize, 2, 8> =
                ImmutableKdTree::new_from_slice(&content_to_add);
        }
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_0() {
        let tree_size = 18;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_1() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_2() {
        let tree_size = 155;
        let seed = 480;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_3() {
        let tree_size = 26; // also 32
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_4() {
        let tree_size = 21;
        let seed = 131851;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_5() {
        let tree_size = 32;
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_6() {
        let tree_size = 56;
        let seed = 450533;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_7() {
        let tree_size = 18;
        let seed = 992063;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_8() {
        let tree_size = 19;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_9() {
        let tree_size = 20;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_10() {
        let tree_size = 36;
        let seed = 375096;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_11() {
        let tree_size = 10000;
        let seed = 257281;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_many_dupes() {
        let tree_size = 8;
        let seed = 0;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let mut duped: Vec<[f32; 4]> = Vec::with_capacity(content_to_add.len() * 10);
        for item in content_to_add {
            for _ in 0..6 {
                duped.push(item);
            }
        }

        let tree: ImmutableKdTree<f32, usize, 4, 8> = ImmutableKdTree::new_from_slice(&duped);

        println!("Tree Stats: {:?}", tree.generate_stats());
    }

    #[ignore]
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
            (0..1_000_000)
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|&seed| {
                    let result = panic::catch_unwind(|| {
                        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                        let content_to_add: Vec<[f32; 4]> =
                            (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

                        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
                            ImmutableKdTree::new_from_slice(&content_to_add);
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
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[ignore]
    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_iterate() {
        let pts = vec![[1.0, 2.0, 3.0], [10.0, 2.0, 3.0], [1.0, 20.0, 3.0]];
        let mut t: ImmutableKdTree<f64, usize, 3, 2> =
            ImmutableKdTree::new_from_slice(pts.as_slice());

        let expected = pts.iter().cloned().enumerate().collect();
        let actual: HashMap<_, _> = t.iter().collect();
        assert_eq!(actual, expected);
    }
}
