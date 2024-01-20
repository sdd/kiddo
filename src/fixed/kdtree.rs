//! Fixed point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are fixed point or integers. [`u8`], [`u16`], [`u32`], and [`u64`] based fixed-point / integers are supported
//! via the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate, eg [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html) for a 16-bit fixed point number with 14 bits after the
//! decimal point.

use az::{Az, Cast};
use divrem::DivCeil;
use fixed::traits::Fixed;
use std::fmt::Debug;
use std::{cmp::PartialEq, collections::VecDeque};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::iter::TreeIter;
use crate::{
    iter::IterableTreeData,
    types::{Content, Index},
};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on [`FixedKdTree`](crate::fixed::kdtree::KdTree). A type from the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed) crate will implement
/// all of the traits required by Axis. For example [`FixedU16<U14>`](https://docs.rs/fixed/1.21.0/fixed/struct.FixedU16.html).
pub trait Axis: Fixed + Default + Debug + Copy + Sync + Send {
    /// used in query methods to update the rd value. Basically a saturating add for Fixed and an add for Float
    fn rd_update(rd: Self, delta: Self) -> Self;
}
impl<T: Fixed + Default + Debug + Copy + Sync + Send> Axis for T {
    #[inline]
    fn rd_update(rd: Self, delta: Self) -> Self {
        rd.saturating_add(delta)
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
/// a [`FixedKdTree`](crate::fixed::kdtree::KdTree). The types in the [`Fixed`](https://docs.rs/fixed/1.21.0/fixed)  crate do not support [`Rkyv`](https://crates.io/crates/rkyv/0.7.39) yet.
/// As a workaround, we need to [`std::mem::transmute`] a [`crate::fixed::kdtree::KdTree`] into
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
    #[cfg_attr(
        feature = "serialize",
        serde(with = "crate::custom_serde::array_of_arrays")
    )]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
    )]
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
    pub(crate) content_points: [[A; K]; B],

    #[cfg_attr(feature = "serialize", serde(with = "crate::custom_serde::array"))]
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

impl<A, T, const K: usize, const B: usize, IDX> Default for KdTree<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
    usize: Cast<IDX>,
{
    fn default() -> Self {
        Self::new()
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
    /// type Fxd = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<Fxd, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// tree.add(&[Fxd::from_num(1), Fxd::from_num(2), Fxd::from_num(5)], 100);
    /// tree.add(&[Fxd::from_num(2), Fxd::from_num(3), Fxd::from_num(6)], 101);
    ///
    /// assert_eq!(tree.size(), 2);
    /// ```
    #[inline]
    pub fn size(&self) -> T {
        self.size
    }

    /// Iterate over all `(index, point)` tuples in arbitrary order.
    ///

    /// ```
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// let point = [1u16, 2, 3];
    /// let tree: KdTree<f16, u32, 3, 32> = KdTree::new();
    /// tree.add(point, 10);
    ///
    /// let mut pairs: Vec<_> = tree.iter().collect()
    /// assert_eq!(pairs.pop(), (10, point));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (T, [A; K])> + '_ {
        TreeIter::new(self)
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    IterableTreeData<A, T, K> for KdTree<A, T, K, B, IDX>
{
    fn get_leaf_data(&self, idx: usize) -> Option<VecDeque<(T, [A; K])>> {
        let leaf = self.leaves.get(idx)?;
        let max = leaf.size.cast();
        Some(
            leaf.content_items
                .iter()
                .cloned()
                .zip(leaf.content_points.iter().cloned())
                .take(max)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use fixed::types::extra::U14;
    use fixed::FixedU16;

    use crate::fixed::kdtree::KdTree;

    type Fxd = FixedU16<U14>;

    #[test]
    fn it_can_be_constructed_with_new() {
        let tree: KdTree<Fxd, u32, 4, 32, u32> = KdTree::new();

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_defined_capacity() {
        let tree: KdTree<Fxd, u32, 4, 32, u32> = KdTree::with_capacity(10);

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_capacity_of_zero() {
        let tree: KdTree<Fxd, u32, 4, 32, u32> = KdTree::with_capacity(0);

        assert_eq!(tree.size(), 0);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn can_serde() {
        let mut tree: KdTree<Fxd, u32, 4, 32, u32> = KdTree::new();

        let content_to_add: [([Fxd; 4], u32); 16] = [
            (
                [
                    Fxd::from_num(0.9),
                    Fxd::from_num(0),
                    Fxd::from_num(0.9),
                    Fxd::from_num(0),
                ],
                9,
            ),
            (
                [
                    Fxd::from_num(0.4),
                    Fxd::from_num(0.5),
                    Fxd::from_num(0.4),
                    Fxd::from_num(0.50),
                ],
                4,
            ),
            (
                [
                    Fxd::from_num(0.12),
                    Fxd::from_num(0.3),
                    Fxd::from_num(0.12),
                    Fxd::from_num(0.3),
                ],
                12,
            ),
            (
                [
                    Fxd::from_num(0.7),
                    Fxd::from_num(0.2),
                    Fxd::from_num(0.7),
                    Fxd::from_num(0.2),
                ],
                7,
            ),
            (
                [
                    Fxd::from_num(0.13),
                    Fxd::from_num(0.4),
                    Fxd::from_num(0.13),
                    Fxd::from_num(0.4),
                ],
                13,
            ),
            (
                [
                    Fxd::from_num(0.6),
                    Fxd::from_num(0.3),
                    Fxd::from_num(0.6),
                    Fxd::from_num(0.3),
                ],
                6,
            ),
            (
                [
                    Fxd::from_num(0.2),
                    Fxd::from_num(0.7),
                    Fxd::from_num(0.2),
                    Fxd::from_num(0.7),
                ],
                2,
            ),
            (
                [
                    Fxd::from_num(0.14),
                    Fxd::from_num(0.5),
                    Fxd::from_num(0.14),
                    Fxd::from_num(0.5),
                ],
                14,
            ),
            (
                [
                    Fxd::from_num(0.3),
                    Fxd::from_num(0.6),
                    Fxd::from_num(0.3),
                    Fxd::from_num(0.6),
                ],
                3,
            ),
            (
                [
                    Fxd::from_num(0.1),
                    Fxd::from_num(0.1),
                    Fxd::from_num(0.10),
                    Fxd::from_num(0.1),
                ],
                10,
            ),
            (
                [
                    Fxd::from_num(0.16),
                    Fxd::from_num(0.7),
                    Fxd::from_num(0.16),
                    Fxd::from_num(0.7),
                ],
                16,
            ),
            (
                [
                    Fxd::from_num(0.1),
                    Fxd::from_num(0.8),
                    Fxd::from_num(0.1),
                    Fxd::from_num(0.8),
                ],
                1,
            ),
            (
                [
                    Fxd::from_num(0.15),
                    Fxd::from_num(0.6),
                    Fxd::from_num(0.15),
                    Fxd::from_num(0.6),
                ],
                15,
            ),
            (
                [
                    Fxd::from_num(0.5),
                    Fxd::from_num(0.4),
                    Fxd::from_num(0.5),
                    Fxd::from_num(0.4),
                ],
                5,
            ),
            (
                [
                    Fxd::from_num(0.8),
                    Fxd::from_num(0.1),
                    Fxd::from_num(0.8),
                    Fxd::from_num(0.1),
                ],
                8,
            ),
            (
                [
                    Fxd::from_num(0.11),
                    Fxd::from_num(0.2),
                    Fxd::from_num(0.11),
                    Fxd::from_num(0.2),
                ],
                11,
            ),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }
        assert_eq!(tree.size(), 16);

        let serialized = serde_json::to_string(&tree).unwrap();
        println!("JSON: {:?}", &serialized);

        let deserialized: KdTree<Fxd, u32, 4, 32, u32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }
}
