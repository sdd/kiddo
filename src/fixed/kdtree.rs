//! Fixed point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are fixed point or integers. [`u8`], [`u16`], [`u32`], and [`u64`] based fixed-point / integers are supported
//! via the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate, eg [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html) for a 16-bit fixed point number with 14 bits after the
//! decimal point.

use az::{Az, Cast};
use divrem::DivCeil;
use fixed::traits::Fixed;
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::types::{Content, Index};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on [`FixedKdTree`](crate::FixedKdTree). A type from the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate will implement
/// all of the traits required by Axis. For example [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html).
pub trait Axis: Fixed + Default + Debug + Copy + Sync {
    /// Returns the maximum value that the type implementing this trait can have
    fn max_value() -> Self;
    /// returns the zero value for this type
    fn zero() -> Self;
    /// used within query functions to update rd from old and new off
    fn rd_update(self, old_off: Self, new_off: Self) -> Self;
}
impl<T: Fixed + Default + Debug + Copy + Sync> Axis for T {
    fn max_value() -> Self {
        Self::MAX
    }

    fn zero() -> Self {
        Self::ZERO
    }

    fn rd_update(self, old_off: Self, new_off: Self) -> Self {
        self.saturating_add(
            (new_off.saturating_mul(new_off)).saturating_sub(old_off.saturating_mul(old_off)),
        )
    }
}

/// Rkyv-serializable equivalent of `kiddo::fixed::kdtree::Axis`
#[cfg(feature = "serialize_rkyv")]
pub trait AxisRK: num_traits::Zero + Default + Debug + rkyv::Archive {}
#[cfg(feature = "serialize_rkyv")]
impl<T: num_traits::Zero + Default + Debug + rkyv::Archive> AxisRK for T {}

/// Rkyv-serializable fixed point k-d tree
///
/// This is only required when using Rkyv to serialize to / deserialize from
/// a [`FixedKdTree`](crate::FixedKdTree). The types in the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed)  crate do not support [`Rkyv`](https://crates.io/crates/rkyv/0.7.39) yet.
/// As a workaround, we need to [`std::mem::transmute`] a [`crate::FixedKdTree`] into
/// an equivalent [`crate::fixed::kdtree::KdTreeRK`] before serializing via Rkyv,
/// and vice-versa when deserializing.
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct KdTreeRK<
    A: num_traits::PrimInt,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
> {
    pub(crate) leaves: Vec<LeafNodeRK<A, T, K, B, IDX>>,
    pub(crate) stems: Vec<StemNodeRK<A, K, IDX>>,
    pub(crate) root_index: IDX,
    pub(crate) size: T,
}

/// Fixed point k-d tree
///
/// For use when the co-ordinates of the points being stored in the tree
/// are fixed point or integers. [`u8`], [`u16`], [`u32`], and [`u64`] based fixed-point / integers are supported
/// via the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate, eg [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html) for a 16-bit fixed point number with 14 bits after the
/// decimal point.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    pub(crate) leaves: Vec<LeafNode<A, T, K, B, IDX>>,
    pub(crate) stems: Vec<StemNode<A, K, IDX>>,
    pub(crate) root_index: IDX,
    pub(crate) size: T,
}

#[doc(hidden)]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct StemNodeRK<A: num_traits::PrimInt, const K: usize, IDX: Index<T = IDX>> {
    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StemNode<A: Copy + Default, const K: usize, IDX> {
    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[doc(hidden)]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct LeafNodeRK<
    A: num_traits::PrimInt,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
> {
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
    pub(crate) content_points: [[A; K]; B],
    pub(crate) content_items: [T; B],
    pub(crate) size: IDX,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LeafNode<
    A: Copy + Default,
    T: Copy + Default,
    const K: usize,
    const B: usize,
    IDX,
> {
    #[cfg_attr(feature = "serialize", serde(with = "array_of_arrays"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
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

impl<A, T, const K: usize, const B: usize, IDX> LeafNode<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
{
    pub(crate) fn new() -> Self {
        Self {
            content_points: [[A::ZERO; K]; B],
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
    /// Creates a new fixed-point/int KdTree.
    ///
    /// Capacity is set by default to 10x the bucket size (32 in this case).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U14;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<FixedU16<U14>, u32, 3, 32, u32> = KdTree::new();
    ///
    /// assert_eq!(tree.size(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        KdTree::with_capacity(B * 10)
    }

    /// Creates a new fixed-point/integer KdTree and reserves capacity for a specific number of items.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U14;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<FixedU16<U14>, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// assert_eq!(tree.size(), 0);
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

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U0;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// type FXD = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 101);
    ///
    /// assert_eq!(tree.size(), 2);
    /// ```
    #[inline]
    pub fn size(&self) -> T {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use fixed::types::extra::U14;
    use fixed::FixedU16;

    use crate::fixed::kdtree::KdTree;

    type FXD = FixedU16<U14>;

    #[test]
    fn it_can_be_constructed_with_new() {
        let tree: KdTree<FXD, u32, 4, 32, u32> = KdTree::new();

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_defined_capacity() {
        let tree: KdTree<FXD, u32, 4, 32, u32> = KdTree::with_capacity(10);

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_capacity_of_zero() {
        let tree: KdTree<FXD, u32, 4, 32, u32> = KdTree::with_capacity(0);

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
