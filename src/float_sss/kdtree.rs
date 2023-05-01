//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently.

use az::{Az, Cast};
use num_traits::Float;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout, Global, Allocator, AllocError};
use std::{cmp::PartialEq};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use divrem::DivCeil;

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::types::{Content, Index};
#[cfg(feature = "serialize")]
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
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    pub(crate) leaves: Vec<LeafNode<A, T, K, B, IDX>>,
    pub(crate) stems: Vec<A>,
    pub(crate) dstems: Vec<StemNode<A, K, IDX>>,
    // pub(crate) root_index: IDX,
    pub(crate) size: T,

    pub(crate) unreserved_leaf_idx: usize,
}


#[doc(hidden)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct StemNode<A: Copy + Default, const K: usize, IDX> {
    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[doc(hidden)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    #[cfg_attr(feature = "serialize", serde(with = "array_of_arrays"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))
    )]
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
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

    pub(crate) size: IDX,
}

impl<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX>
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
        let stem_capacity = (leaf_capacity - 1).max(1);

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
            size: T::zero(),
            stems,
            dstems: Vec::with_capacity(0),
            leaves,
            // root_index: <IDX as Index>::leaf_offset(),
            unreserved_leaf_idx: leaf_capacity
        };

        tree.leaves[0].size = IDX::zero();
        tree.stems[0] = A::nan();

        tree
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
    pub fn size(&self) -> T {
        self.size
    }

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
    use crate::float::kdtree::KdTree;
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
    // #[cfg(feature = "serialize")]
    // #[test]
    // fn can_serde() {
    //     let mut tree: KdTree<u16, u32, 4, 32, u32> = KdTree::new();
    //
    //     let content_to_add: [(PT, T); 16] = [
    //         ([9f32, 0f32, 9f32, 0f32], 9),
    //         ([4f32, 500f32, 4f32, 500f32], 4),
    //         ([12f32, -300f32, 12f32, -300f32], 12),
    //         ([7f32, 200f32, 7f32, 200f32], 7),
    //         ([13f32, -400f32, 13f32, -400f32], 13),
    //         ([6f32, 300f32, 6f32, 300f32], 6),
    //         ([2f32, 700f32, 2f32, 700f32], 2),
    //         ([14f32, -500f32, 14f32, -500f32], 14),
    //         ([3f32, 600f32, 3f32, 600f32], 3),
    //         ([10f32, -100f32, 10f32, -100f32], 10),
    //         ([16f32, -700f32, 16f32, -700f32], 16),
    //         ([1f32, 800f32, 1f32, 800f32], 1),
    //         ([15f32, -600f32, 15f32, -600f32], 15),
    //         ([5f32, 400f32, 5f32, 400f32], 5),
    //         ([8f32, 100f32, 8f32, 100f32], 8),
    //         ([11f32, -200f32, 11f32, -200f32], 11),
    //     ];
    //
    //     for (point, item) in content_to_add {
    //         tree.add(&point, item);
    //     }
    //     assert_eq!(tree.size(), 16);
    //
    //     let serialized = serde_json::to_string(&tree).unwrap();
    //     println!("JSON: {:?}", &serialized);
    //
    //     let deserialized: KdTree = serde_json::from_str(&serialized).unwrap();
    //     assert_eq!(tree, deserialized);
    // }
}
