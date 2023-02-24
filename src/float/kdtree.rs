//! Floating point KD Tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently.

use az::{Az, Cast};
use num_traits::Float;
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::float::util::{distance_to_bounds, extend};
use crate::types::{Content, Index};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on the float `KdTree`. This will be `f64` or `f32`.
pub trait Axis: Float + Default + Debug + Copy + Sync {}
impl<T: Float + Default + Debug + Copy + Sync> Axis for T {}

/// Floating point KD Tree
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
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) min_bound: [A; K],
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) max_bound: [A; K],

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
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de>"
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

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) min_bound: [A; K],

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) max_bound: [A; K],

    pub(crate) size: IDX,
}

impl<A, const K: usize, IDX> StemNode<A, K, IDX>
where
    A: Axis,
    IDX: Index<T = IDX>
{
    pub(crate) fn extend(&mut self, point: &[A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }
}

impl<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX>
    LeafNode<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>
{
    pub(crate) fn new() -> Self {
        Self {
            min_bound: [A::max_value(); K],
            max_bound: [A::min_value(); K],
            content_points: [[A::zero(); K]; B],
            content_items: [T::zero(); B],
            size: IDX::zero(),
        }
    }

    pub(crate) fn extend(&mut self, point: &[A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }
}

impl<A, T, const K: usize, const B: usize, IDX>
    KdTree<A, T, K, B, IDX>
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
    /// use kiddo::KdTree;
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
            stems: Vec::with_capacity(capacity.ilog2() as usize),
            leaves: Vec::with_capacity(capacity.div_ceil(B.az::<usize>())),
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
    /// use kiddo::KdTree;
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

    pub(crate) fn child_dist_to_bounds<F>(
        &self,
        query: &[A; K],
        child_node_idx: IDX,
        distance_fn: &F,
    ) -> A
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(child_node_idx) {
            distance_to_bounds(
                query,
                &self.stems[child_node_idx.az::<usize>()].min_bound,
                &self.stems[child_node_idx.az::<usize>()].max_bound,
                distance_fn,
            )
        } else {
            distance_to_bounds(
                query,
                &self.leaves[(child_node_idx - IDX::leaf_offset()).az::<usize>()].min_bound,
                &self.leaves[(child_node_idx - IDX::leaf_offset()).az::<usize>()].max_bound,
                distance_fn,
            )
        }
    }
}

// impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index> Default for KdTree<A, T, K, B, IDX> where usize: Cast<IDX> {
//     fn default() -> Self {
//         Self::new()
//     }
// }

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
