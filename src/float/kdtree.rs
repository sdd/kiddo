//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently.

use az::{Az, Cast};
use divrem::DivCeil;
use num_traits::Float;
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::types::{Content, Index};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on the float [`KdTree`]. This will be [`f64`] or [`f32`].
pub trait Axis: Float + Default + Debug + Copy + Sync {
    /// returns absolute diff between two values of a type implementing this trait
    fn saturating_dist(self, other: Self) -> Self;

    /// used within query functions to update rd from old and new off
    fn rd_update(self, old_off: Self, new_off: Self) -> Self;
}
impl<T: Float + Default + Debug + Copy + Sync> Axis for T {
    fn saturating_dist(self, other: Self) -> Self {
        (self - other).abs()
    }

    fn rd_update(self, old_off: Self, new_off: Self) -> Self {
        self + new_off * new_off - old_off * old_off
    }
}

// TODO: make LeafNode and StemNode `pub(crate)` so that they,
//       and their Archived types, don't show up in docs.
//       This is tricky due to encountering this problem:
//       https://github.com/rkyv/rkyv/issues/275
/* #[cfg_attr(
    feature = "serialize_rkyv",
    omit_bounds
)] */

/// Floating point k-d tree
///
/// For use when the co-ordinates of the points being stored in the tree
/// on the float [`KdTree`]. This will be [`f64`] or [`f32`].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    pub(crate) leaves: Vec<LeafNode<A, T, K, B, IDX>>,
    pub(crate) stems: Vec<StemNode<A, K, IDX>>,
    pub(crate) root_index: IDX,
    pub(crate) size: T,
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
        KdTree::with_capacity(B * 10)
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
        let mut tree = Self {
            size: T::zero(),
            stems: Vec::with_capacity(capacity.max(1).ilog2() as usize),
            leaves: Vec::with_capacity(DivCeil::div_ceil(capacity, B.az::<usize>())),
            root_index: <IDX as Index>::leaf_offset(),
        };

        tree.leaves.push(LeafNode::new());

        tree
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

macro_rules! generate_common_methods {
    ($kdtree:ident) => {
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
    };
}

impl<A, T, const K: usize, const B: usize, IDX> KdTree<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
    usize: Cast<IDX>,
{
    generate_common_methods!(KdTree);
}

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
    generate_common_methods!(ArchivedKdTree);
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
