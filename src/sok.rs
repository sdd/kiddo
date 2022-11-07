use crate::util::{distance_to_bounds, extend};
// use crate::bounds_extender::BoundsExtender;
use num_traits::Float;
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;

pub(crate) const LEAF_OFFSET: usize = usize::MAX.overflowing_shr(1).0;

pub trait Axis: Float + Default + Debug {}
impl<T: Float + Default + Debug> Axis for T {}

pub trait Content: PartialEq + Default + Clone + Copy + Ord + Debug {}
impl<T: PartialEq + Default + Clone + Copy + Ord + Debug> Content for T {}

// A: Axis, ie points
// T: Content
// K: Dimensions
// B: Bucket size
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdTree<A: Axis, T: Content, const K: usize, const B: usize> {
    pub(crate) size: usize,
    pub(crate) root_index: usize,

    pub stems: Vec<StemNode<A, K>>,
    pub leaves: Vec<LeafNode<A, T, K, B>>,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StemNode<A: Axis, const K: usize> {
    // TODO: investigate changing usize to u32
    pub(crate) left: usize,
    pub(crate) right: usize,

    pub(crate) split_val: A,

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) min_bound: [A; K],
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) max_bound: [A; K],
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafNode<A: Axis, T: Content, const K: usize, const B: usize> {
    pub(crate) size: usize,

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) content: [LeafNodeEntry<A, T, K>; B],

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
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LeafNodeEntry<A: Axis, T: Content, const K: usize> {
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) point: [A; K],
    pub(crate) item: T,
}

impl<A: Axis, T: Content, const K: usize> Default for LeafNodeEntry<A, T, K> {
    fn default() -> Self {
        LeafNodeEntry {
            point: [A::zero(); K],
            item: T::default(),
        }
    }
}

impl<A: Axis, T: Content, const K: usize> LeafNodeEntry<A, T, K> {
    pub(crate) fn new(point: [A; K], item: T) -> Self {
        LeafNodeEntry { point, item }
    }
}

impl<A: Axis, const K: usize> StemNode<A, K> {
    pub(crate) fn extend(&mut self, point: &[A; K]) {
        /*BoundsExtender::*/
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize> LeafNode<A, T, K, B> {
    pub(crate) fn new() -> Self {
        Self {
            size: 0,
            content: [LeafNodeEntry::default(); B],
            min_bound: [A::infinity(); K],
            max_bound: [A::neg_infinity(); K],
        }
    }

    pub(crate) fn extend(&mut self, point: &[A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }

    pub(crate) fn extend_with_result(&mut self, point: &[A; K]) -> ([A; K], [A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
        return (self.min_bound, self.max_bound);
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize> KdTree<A, T, K, B> {
    #[inline]
    pub fn new() -> Self {
        KdTree::with_capacity(B * 10)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut tree = Self {
            size: 0,
            stems: Vec::with_capacity(capacity.ilog2() as usize),
            leaves: Vec::with_capacity(capacity.div_ceil(B)),
            root_index: LEAF_OFFSET,
        };

        tree.leaves.push(LeafNode::new());

        tree
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn is_stem_index(x: usize) -> bool {
        x < LEAF_OFFSET
    }

    pub(crate) fn child_dist_to_bounds<F>(
        &self,
        query: &[A; K],
        child_node_idx: usize,
        distance_fn: &F,
    ) -> A
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B>::is_stem_index(child_node_idx) {
            distance_to_bounds(
                query,
                &self.stems[child_node_idx].min_bound,
                &self.stems[child_node_idx].max_bound,
                distance_fn,
            )
        } else {
            distance_to_bounds(
                query,
                &self.leaves[child_node_idx - LEAF_OFFSET].min_bound,
                &self.leaves[child_node_idx - LEAF_OFFSET].max_bound,
                distance_fn,
            )
        }
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize> Default for KdTree<A, T, K, B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::KdTree;

    #[test]
    fn it_can_be_constructed_with_new() {
        let tree: KdTree<f64, i32, 2, 10> = KdTree::new();

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_defined_capacity() {
        let tree: KdTree<f64, i32, 2, 10> = KdTree::with_capacity(10);

        assert_eq!(tree.size(), 0);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn can_serde() {
        let mut tree: KdTree<f64, usize, 2, 4> = KdTree::new();

        let content_to_add = [
            ([9f64, 0f64], 9),
            ([4f64, 500f64], 4),
            ([12f64, -300f64], 12),
            ([7f64, 200f64], 7),
            ([13f64, -400f64], 13),
            ([6f64, 300f64], 6),
            ([2f64, 700f64], 2),
            ([14f64, -500f64], 14),
            ([3f64, 600f64], 3),
            ([10f64, -100f64], 10),
            ([16f64, -700f64], 16),
            ([1f64, 800f64], 1),
            ([15f64, -600f64], 15),
            ([5f64, 400f64], 5),
            ([8f64, 100f64], 8),
            ([11f64, -200f64], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }
        assert_eq!(tree.size(), 16);

        let serialized = serde_json::to_string(&tree).unwrap();
        println!("JSON: {:?}", &serialized);

        let deserialized: KdTree<f64, usize, 2, 4> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tree, deserialized);
    }
}
