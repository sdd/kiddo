//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently.

use az::{Az, Cast};
use divrem::DivCeil;
use num_traits::Float;
use ordered_float::OrderedFloat;
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::Rem;

#[cfg(feature = "serde")]
use crate::custom_serde::*;
use crate::types::{Content, Index};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait FloatLSB {
    fn is_lsb_set(self) -> bool;
    fn with_lsb_set(self) -> Self;
    fn with_lsb_clear(self) -> Self;
    fn is_2lsb_set(self) -> bool;
    fn with_2lsb_set(self) -> Self;
    fn with_2lsb_clear(self) -> Self;
}

impl FloatLSB for f32 {
    fn is_lsb_set(self) -> bool {
        self.to_bits() & 1u32 != 0
    }

    fn with_lsb_set(self) -> f32 {
        f32::from_bits(self.to_bits() | 1u32)
    }

    fn with_lsb_clear(self) -> f32 {
        f32::from_bits(self.to_bits() & 0xFFFFFFFE)
    }

    fn is_2lsb_set(self) -> bool {
        self.to_bits() & 2u32 != 0
    }

    fn with_2lsb_set(self) -> f32 {
        f32::from_bits(self.to_bits() | 2u32)
    }

    fn with_2lsb_clear(self) -> f32 {
        f32::from_bits(self.to_bits() & 0xFFFFFFFD)
    }
}

impl FloatLSB for f64 {
    fn is_lsb_set(self) -> bool {
        self.to_bits() & 1u64 != 0
    }

    fn with_lsb_set(self) -> f64 {
        f64::from_bits(self.to_bits() | 1u64)
    }

    fn with_lsb_clear(self) -> f64 {
        f64::from_bits(self.to_bits() & 0xFFFFFFFFFFFFFFFE)
    }

    fn is_2lsb_set(self) -> bool {
        self.to_bits() & 2u64 != 0
    }

    fn with_2lsb_set(self) -> f64 {
        f64::from_bits(self.to_bits() | 2u64)
    }

    fn with_2lsb_clear(self) -> f64 {
        f64::from_bits(self.to_bits() & 0xFFFFFFFFFFFFFFFD)
    }
}

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on the float `KdTree`. This will be `f64` or `f32`.
pub trait Axis: Float + Default + Debug + Copy + Sync + FloatLSB {}
impl<T: Float + Default + Debug + Copy + Sync + FloatLSB> Axis for T {}

/// Floating point k-d tree
///
/// For use when the co-ordinates of the points being stored in the tree
/// are floats. f64 or f32 are supported currently
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    pub(crate) leaves: Vec<LeafNode<A, T, K, B, IDX>>,
    pub(crate) stems: Vec<A>,
    pub(crate) dstems: Vec<StemNode<A, K, IDX>>,
    pub(crate) size: usize,

    pub(crate) unreserved_leaf_idx: usize,
    pub(crate) optimized_read_only: bool,
}

#[doc(hidden)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct StemNode<A: Copy + Default, const K: usize, IDX> {
    // pub(crate) left: IDX,
    // pub(crate) right: IDX,
    pub(crate) children: [IDX; 2],
    pub(crate) split_val: A,
}

#[doc(hidden)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    #[cfg_attr(feature = "serde", serde(with = "array_of_arrays"))]
    #[cfg_attr(
        feature = "serde",
        serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))
    )]
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
    pub(crate) content_points: [[A; K]; B],

    #[cfg_attr(feature = "serde", serde(with = "array"))]
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) content_items: [T; B],

    pub(crate) size: IDX,
}

/* impl<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX>
    LeafNode<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
{
    pub(crate) fn new() -> Self {
        Self {
            content_points: [[A::zero(); K]; B],
            content_items: [T::zero(); B],
            size: IDX::zero(),
        }
    }
} */

#[allow(dead_code)]
#[derive(Debug)]
pub struct TreeStats {
    dstem_node_count: usize,
    leaf_fill_counts: Vec<usize>,
    leaf_fill_ratio: f32,
    stem_fill_ratio: f32,
    unused_stem_count: usize,
}

impl<A, T, const K: usize, const B: usize, IDX> KdTree<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
    usize: Cast<IDX>,
{
    /// Creates a new float KdTree.
    ///
    /// Capacity is set by default to 10x the bucket size (32 in this case).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn new() -> Self {
        KdTree::with_capacity(B * 16)
    }

    /// Creates a new float KdTree and reserve capacity for a specific number of items.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity <= <IDX as Index>::capacity_with_bucket_size(B));

        let leaf_capacity = DivCeil::div_ceil(capacity, B.az::<usize>()).next_power_of_two();
        let stem_capacity = leaf_capacity.max(1);

        let layout = Layout::array::<A>(stem_capacity).unwrap();
        let stems = unsafe {
            let mem = match Global.allocate(layout) {
                Ok(mem) => mem.cast::<A>().as_ptr(),
                Err(AllocError) => panic!(),
            };

            Vec::from_raw_parts_in(mem, stem_capacity, stem_capacity, Global)
        };

        let layout = Layout::array::<LeafNode<A, T, K, B, IDX>>(leaf_capacity).unwrap();
        let leaves = unsafe {
            let mem = match Global.allocate(layout) {
                Ok(mem) => mem.cast::<LeafNode<A, T, K, B, IDX>>().as_ptr(),
                Err(AllocError) => panic!(),
            };

            Vec::from_raw_parts_in(mem, leaf_capacity, leaf_capacity, Global)
        };

        let mut tree = Self {
            size: 0,
            stems,
            dstems: Vec::with_capacity(0),
            leaves,
            unreserved_leaf_idx: leaf_capacity,
            optimized_read_only: false,
        };

        tree.leaves[0].size = IDX::zero();

        // Set this to infinity so that if it is accessed, things will break
        tree.stems[0] = A::infinity();

        // 1 is the true root, so that we can use *2 and *2+1 to traverse down
        tree.stems[1] = A::nan();

        tree
    }

    /// Creates a new float KdTree, balanced and optimized.
    ///
    /// Trees constructed using this method will not be modifiable
    /// after construction, ant will be optimally balanced and tuned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!();
    /// let tree: KdTree<f64, u32, 3, 32, u32> = KdTree::optimize_from(points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn optimize_from(source: &[[A; K]]) -> Self
    where
        usize: Cast<T>,
    {
        let item_count = source.len();
        let mut leaf_node_count = item_count.div_ceil(B);
        let mut stem_node_count = leaf_node_count.next_power_of_two();

        let mut stems;
        let mut leaves;

        let mut total_shift = 0;
        loop {
            leaves = Self::allocate_leaves(leaf_node_count);

            stems = vec![A::infinity(); stem_node_count];
            let mut sort_index = Vec::from_iter(0..item_count);
            stems[0] = A::infinity();

            let requested_shift =
                Self::optimize_stems(&mut stems, source, &mut sort_index, 1, 0, total_shift);

            if requested_shift == 0 {
                break;
            }
            total_shift += requested_shift;

            stem_node_count = (stem_node_count + 1).next_power_of_two();
            leaf_node_count += requested_shift.div_ceil(B);
        }

        let mut tree = Self {
            size: 0,
            stems,
            dstems: Vec::with_capacity(0),
            leaves,
            unreserved_leaf_idx: leaf_node_count,
            optimized_read_only: true,
        };

        for (idx, point) in source.iter().enumerate() {
            tree.add_to_optimized(point, idx.az::<T>());
        }

        tree
    }

    fn allocate_leaves(count: usize) -> Vec<LeafNode<A, T, K, B, IDX>> {
        let layout = Layout::array::<LeafNode<A, T, K, B, IDX>>(count).unwrap();
        let mut leaves = unsafe {
            let mem = match Global.allocate(layout) {
                Ok(mem) => mem.cast::<LeafNode<A, T, K, B, IDX>>().as_ptr(),
                Err(AllocError) => panic!(),
            };

            Vec::from_raw_parts_in(mem, count, count, Global)
        };
        for leaf in &mut leaves {
            leaf.size = IDX::zero();
        }

        leaves
    }

    /**
     *  1234456789ABCDEF: Initial alloc
     *
     *  12344567 89ABCDEF: First split
     *
     *  1234 4567        : Second split
     *
     *  123 44567        :  Second split adjust: overflow
     *
     *  1234456 789ABCDE F : desired readjustment
     *
     *  123 4456           : Revised second split
     */

    /// the value returned is zero if balancing was successful. If however a child splitpoint has
    /// landed in between two (or more) items with the same value, the value returned is a hint to
    /// the caller of how many items overflowed out of the second bucket due to the optimization
    /// attempt overflowing after the pivot was moved.
    fn optimize_stems(
        stems: &mut Vec<A>,
        source: &[[A; K]],
        sort_index: &mut [usize],
        stem_index: usize,
        dim: usize,
        shifted: usize,
    ) -> usize {
        let next_dim = (dim + 1).rem(K);

        // TODO: should this have one subtracted from it?
        let stem_levels_below = stems.len().ilog2() - stem_index.ilog2() - 1;

        // TODO: this is wrong, it assumes that there is a leaf for every stem
        let items_below = 2usize.pow(stem_levels_below + 1) * B;

        // If there are few enough items that we could fit all of them in the left subtree,
        // leave the current stem val as +inf to push everything down into the left and
        // recurse down without splitting.
        if sort_index.len() + shifted <= items_below / 2 {
            if sort_index.len() > B {
                Self::optimize_stems(
                    stems,
                    source,
                    sort_index,
                    stem_index << 1,
                    next_dim,
                    shifted,
                );
            }
            return 0;
        }

        let mut pivot = (sort_index.len() + shifted) >> 1;
        if stem_index == 1 {
            // If at the top level, check if there's been a shift
            pivot = if shifted > 0 {
                // if so,
                sort_index.len()
            } else {
                // otherwise, do a special case pivot shift to ensure the left subtree is full.
                //(pivot - (B / 2) + 1).next_power_of_two()
                if sort_index.len() & 1 == 1 {
                    (pivot + 1).next_power_of_two()
                } else {
                    pivot.next_power_of_two()
                }
            };
        } else if sort_index.len() & 0x01 == 1 && shifted == 0 {
            pivot = (pivot + 1).next_power_of_two()
        } else {
            pivot = pivot.next_power_of_two();
        }
        pivot = pivot - shifted;
        let orig_pivot = pivot;

        // let max_sort_index = (pivot + 1).min(sort_index.len() - 1);
        sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));

        // this is overkill. We can select_nth_unstable_by_key of (&mut sort_index[..max_sort_index]) for max_sort_index - 1
        // and then, if we get inside the while loop below, keep call it again on a slice of 1 smaller
        // (&mut sort_index[..max_sort_index]).sort_by_cached_key(|&i| OrderedFloat(source[i][dim]));

        (&mut sort_index[..pivot])
            .select_nth_unstable_by_key(pivot - 1, |&i| OrderedFloat(source[i][dim]));

        // if the pivot straddles two values that are equal,
        // keep nudging it left until they aren't
        while sort_index.len() > 1
            && source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim]
            && pivot > 0
        {
            pivot -= 1;
            (&mut sort_index[..pivot])
                .select_nth_unstable_by_key(pivot - 1, |&i| OrderedFloat(source[i][dim]));
        }

        // if we end up with a pivot of 0, something has gone wrong,
        // unless we only had a slice of len 1 anyway
        debug_assert!(pivot > 0 || sort_index.len() == 1);

        // if we have had to nudge left, abort early with non-zero to instruct parent to rebalance
        if pivot < orig_pivot {
            return orig_pivot - pivot;
        }

        stems[stem_index] = source[sort_index[pivot]][dim];

        // if the total number of items that we have to the left of the pivot can fit
        // in a single bucket, we're done
        if pivot <= B {
            return 0;
        }

        // are we on the bottom row? Recursion termination case
        if stem_levels_below == 0 {
            // if the right bucket will overflow, return the overflow amount
            if (source.len() - pivot) > B {
                return (source.len() - pivot) - B;
            }

            // if the right bucket won't overflow, we're good.
            return 0;
        }

        let next_stem_index = stem_index << 1;
        let mut requested_shift_amount;
        let mut shift = 0;
        let mut lower_sort_index;
        let mut upper_sort_index;
        loop {
            (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);

            requested_shift_amount = Self::optimize_stems(
                stems,
                source,
                lower_sort_index,
                next_stem_index,
                next_dim,
                shift,
            );

            if requested_shift_amount == 0 {
                break;
            } else {
                pivot -= requested_shift_amount;
                stems[stem_index] = source[sort_index[pivot]][dim];
                shift += requested_shift_amount;

                // Test for RHS now having more items than can fit
                // in the buckets present in its subtree. If it does,
                // return with a value so that the parent reduces our
                // total allocation
                let new_upper_size = sort_index.len() - pivot;
                if new_upper_size > items_below >> 1 {
                    return new_upper_size - (items_below >> 1);
                }
            }
        }

        let next_stem_index = next_stem_index + 1;
        Self::optimize_stems(
            stems,
            source,
            upper_sort_index,
            next_stem_index,
            next_dim,
            0,
        )
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

    #[inline]
    pub fn capacity(&self) -> usize {
        self.leaves.len() * B
    }

    #[inline]
    pub(crate) fn is_stem_index(x: IDX) -> bool {
        x < <IDX as Index>::leaf_offset()
    }

    pub(crate) fn initialise_dstems(&mut self) {
        let leaf_capacity = self.leaves.capacity();

        let layout = Layout::array::<StemNode<A, K, IDX>>(leaf_capacity).unwrap();
        self.dstems = unsafe {
            let mem = match Global.allocate(layout) {
                Ok(mem) => mem.cast::<StemNode<A, K, IDX>>().as_ptr(),
                Err(AllocError) => panic!(),
            };

            Vec::from_raw_parts_in(mem, leaf_capacity, leaf_capacity, Global)
        };
    }

    pub fn generate_stats(&self) -> TreeStats {
        let mut leaf_fill_counts = vec![0usize; B + 1];
        for leaf in &self.leaves {
            leaf_fill_counts[leaf.size.az::<usize>()] += 1;
        }

        let leaf_fill_ratio = (self.size as f32) / (self.capacity() as f32);

        let unused_stem_count = self.stems.iter().filter(|x| x.is_infinite()).count() - 1;

        let stem_fill_ratio = 1.0 - (unused_stem_count as f32 / ((self.stems.len() - 1) as f32));

        let dstem_node_count = self.dstems.len();

        TreeStats {
            dstem_node_count,
            leaf_fill_counts,
            leaf_fill_ratio,
            stem_fill_ratio,
            unused_stem_count,
        }
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> From<&Vec<[A; K]>>
    for KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    usize: Cast<T>,
{
    fn from(vec: &Vec<[A; K]>) -> Self {
        let mut tree: KdTree<A, T, K, B, IDX> = KdTree::with_capacity(vec.len());

        vec.iter().enumerate().for_each(|(idx, pos)| {
            tree.add(pos, idx.az::<T>());
        });

        tree
    }
}

#[cfg(test)]
mod tests {
    use crate::float_sss::kdtree::KdTree;
    use num_traits::Pow;
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};
    use std::panic;
    type AX = f64;

    #[test]
    fn it_can_be_constructed_with_new() {
        let tree: KdTree<AX, u32, 4, 32, u32> = KdTree::new();

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_defined_capacity() {
        let tree: KdTree<AX, u32, 4, 32, u32> = KdTree::with_capacity(10);

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_capacity_of_zero() {
        let tree: KdTree<AX, u32, 4, 32, u32> = KdTree::with_capacity(0);

        assert_eq!(tree.size(), 0);
    }
    /*    #[cfg(feature = "serde")]
    #[test]
    fn can_serde() {
        let mut tree: KdTree<u16, u32, 4, 32, u32> = KdTree::new();

        let content_to_add: [(PT, T); 16] = [
            ([9f32, 0f32, 9f32, 0f32], 9),
            ([4f32, 500f32, 4f32, 500f32], 4),
            ([12f32, -300f32, 12f32, -300f32], 12),
            ([7f32, 200f32, 7f32, 200f32], 7),
            ([13f32, -400f32, 13f32, -400f32], 13),
            ([6f32, 300f32, 6f32, 300f32], 6),
            ([2f32, 700f32, 2f32, 700f32], 2),
            ([14f32, -500f32, 14f32, -500f32], 14),
            ([3f32, 600f32, 3f32, 600f32], 3),
            ([10f32, -100f32, 10f32, -100f32], 10),
            ([16f32, -700f32, 16f32, -700f32], 16),
            ([1f32, 800f32, 1f32, 800f32], 1),
            ([15f32, -600f32, 15f32, -600f32], 15),
            ([5f32, 400f32, 5f32, 400f32], 5),
            ([8f32, 100f32, 8f32, 100f32], 8),
            ([11f32, -200f32, 11f32, -200f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }
        assert_eq!(tree.size(), 16);

        let serialized = serde_json::to_string(&tree).unwrap();
        println!("JSON: {:?}", &serialized);

        let deserialized: KdTree = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }*/

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

        let tree: KdTree<f32, usize, 2, 4, u32> = KdTree::optimize_from(&content_to_add);

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

        let tree: KdTree<f32, usize, 2, 4, u32> = KdTree::optimize_from(&content_to_add);

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

        let tree: KdTree<f32, usize, 2, 4, u32> = KdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats());

        assert_eq!(tree.leaves[0].size, 3);
        assert_eq!(tree.leaves[1].size, 4);
        assert_eq!(tree.leaves[2].size, 4);
        assert_eq!(tree.leaves[3].size, 4);
        assert_eq!(tree.leaves[4].size, 4);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: KdTree<f32, usize, 4, 4, u32> = KdTree::optimize_from(&content_to_add);
    }

    #[ignore]
    #[test]
    fn can_construct_optimized_tree_multi_rand_increasing_size() {
        let mut failed = false;

        for tree_size in 16..=1024 {
            for seed in 0..100 {
                let result = panic::catch_unwind(|| {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                    let content_to_add: Vec<[f32; 4]> =
                        (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

                    let _tree: KdTree<f32, usize, 4, 4, u32> =
                        KdTree::optimize_from(&content_to_add);
                });

                if result.is_err() {
                    failed = true;
                    println!("Failed on tree size {}, seed #{}", tree_size, seed);
                }
            }
        }

        assert!(!failed);
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

        let tree: KdTree<f32, usize, 4, 4, u32> = KdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }

    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        // let num_uniq = content_to_add
        //     .iter()
        //     .flatten()
        //     .map(|&x| OrderedFloat(x))
        //     .unique()
        //     .count();

        let tree: KdTree<f32, usize, 4, 32, u32> = KdTree::optimize_from(&content_to_add);

        println!("Tree Stats: {:?}", tree.generate_stats())
    }
}
