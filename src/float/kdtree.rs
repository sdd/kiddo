//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently.

use az::{Az, Cast};
use divrem::DivCeil;
use num_traits::float::FloatCore;
use std::fmt::Debug;
use std::{cmp::PartialEq};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::{
    iter::{IterableTreeData, TreeIter},
    types::{Content, Index},
};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on the float [`KdTree`]. This will be [`f64`] or [`f32`].
pub trait Axis: FloatCore + Default + Debug + Copy + Sync + Send + std::ops::AddAssign {
    /// returns absolute diff between two values of a type implementing this trait
    fn saturating_dist(self, other: Self) -> Self;

    /// used in query methods to update the rd value. Basically a saturating add for Fixed and an add for Float
    fn rd_update(rd: Self, delta: Self) -> Self;
}
impl<T: FloatCore + Default + Debug + Copy + Sync + Send + std::ops::AddAssign> Axis for T {
    fn saturating_dist(self, other: Self) -> Self {
        (self - other).abs()
    }

    #[inline]
    fn rd_update(rd: Self, delta: Self) -> Self {
        rd + delta
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
///
/// A convenient type alias exists for KdTree with some sensible defaults set: [`kiddo::KdTree`](`crate::KdTree`).

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
    #[cfg_attr(
        feature = "serialize",
        serde(with = "crate::custom_serde::array_of_arrays")
    )]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>"))
    )]
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
    pub content_points: [[A; K]; B],

    #[cfg_attr(feature = "serialize", serde(with = "crate::custom_serde::array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub content_items: [T; B],

    pub size: IDX,
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
    /// Creates a new float KdTree.
    ///
    /// Capacity is set by default to 10x the bucket size (32 in this case).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, 3> = KdTree::new();
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
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, 3> = KdTree::with_capacity(1_000_000);
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

    /// Iterate over all `(index, point)` tuples in arbitrary order.
    ///

    /// ```
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let point = [1.0f64, 2.0f64, 3.0f64];
    /// let tree: KdTree<f64, u32, 3, 32> = KdTree::new();
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
    fn get_leaf_data(&self, idx: usize, out: &mut Vec<(T, [A; K])>) -> Option<usize> {
        let leaf = self.leaves.get(idx)?;
        let max = leaf.size.cast();
        out.extend(
            leaf.content_items
                .iter()
                .cloned()
                .zip(leaf.content_points.iter().cloned())
                .take(max),
        );
        Some(max)
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
        /// use kiddo::KdTree;
        ///
        /// let mut tree: KdTree<f64, 3> = KdTree::new();
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
    use std::collections::HashMap;

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

    #[cfg(feature = "serialize")]
    #[test]
    fn can_serde() {
        let mut tree: KdTree<f32, u32, 4, 32, u32> = KdTree::new();

        let content_to_add: [([f32; 4], u32); 16] = [
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

        let deserialized: KdTree<f32, u32, 4, 32, u32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }

    #[test]
    fn can_iterate() {
        let mut t: KdTree<f64, i32, 3, 2, u16> = KdTree::new();
        let expected: HashMap<_, _> = vec![
            (10, [1.0, 2.0, 3.0]),
            (12, [10.0, 2.0, 3.0]),
            (15, [1.0, 20.0, 3.0]),
        ]
        .into_iter()
        .collect();

        for (k, v) in expected.iter() {
            t.add(v, *k);
        }
        let actual: HashMap<_, _> = t.iter().collect();
        assert_eq!(actual, expected);
    }
}
