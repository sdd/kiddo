use az::{Az, Cast};
use std::cmp::PartialEq;
use std::fmt::Debug;
use fixed::traits::Fixed;
use num_traits::{One, PrimInt, Unsigned, Zero};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::fixed::util::{distance_to_bounds, extend};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

pub trait Axis: Fixed + Default + Debug + Copy {}
impl<T: Fixed + Default + Debug + Copy> Axis for T {}

#[cfg(feature = "serialize_rkyv")]
pub trait AxisRK: Zero + Default + Debug + rkyv::Archive {}
#[cfg(feature = "serialize_rkyv")]
impl<T: Zero + Default + Debug + rkyv::Archive> AxisRK for T {}

pub trait Content: Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug {}
impl<T: Zero + One + PartialEq + Default + Clone + Copy + Ord + Debug> Content for T {}

pub trait Index: PrimInt + Unsigned + Zero + Cast<usize> {
    type T: Cast<usize>;
    fn max() -> Self;
    fn min() -> Self;
    fn leaf_offset() -> Self;
    fn ilog2(self) -> Self;
    fn div_ceil(self, b: Self::T) -> Self;
}

impl Index for u32 {
    type T = u32;
    fn max() -> u32 {
        u32::MAX
    }
    fn min() -> u32 {
        0u32
    }
    fn leaf_offset() -> u32 {
        u32::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u32 { u32::ilog2(self) }
    fn div_ceil(self, b: u32) -> u32 { u32::div_ceil(self, b) }
}
impl Index for u16 {
    type T = u16;
    fn max() -> u16 {
        u16::MAX
    }
    fn min() -> u16 {
        0u16
    }
    fn leaf_offset() -> u16 {
        u16::MAX.overflowing_shr(1).0
    }
    fn ilog2(self) -> u16 { u16::ilog2(self) as u16 }
    fn div_ceil(self, b: u16) -> u16 { u16::div_ceil(self, b) }
}

#[cfg_attr(
feature = "serialize_rkyv",
derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct KdTreeRK<A: PrimInt, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> {
    pub leaves: Vec<LeafNodeRK<A, T, K, B, IDX>>,
    pub stems: Vec<StemNodeRK<A, K, IDX>>,
    pub(crate) root_index: IDX,
    size: T,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> {
    pub leaves: Vec<LeafNode<A, T, K, B, IDX>>,
    pub stems: Vec<StemNode<A, K, IDX>>,
    pub(crate) root_index: IDX,
    pub(crate) size: T,
}

#[cfg_attr(
feature = "serialize_rkyv",
derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct StemNodeRK<A: PrimInt, const K: usize, IDX: Index<T = IDX>> {
    pub(crate) min_bound: [A; K],
    pub(crate) max_bound: [A; K],

    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct StemNode<A: Axis, const K: usize, IDX: Index<T = IDX>> {
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) min_bound: [A; K],
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) max_bound: [A; K],

    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[cfg_attr(
feature = "serialize_rkyv",
derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg(feature = "serialize_rkyv")]
pub struct LeafNodeRK<A: PrimInt, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> {
    // TODO: Refactor content_points to be [[A; B]; K] to see if this helps vectorisation
    pub(crate) content_points: [[A; K]; B],
    pub(crate) content_items: [T; B],
    pub(crate) min_bound: [A; K],
    pub(crate) max_bound: [A; K],
    pub(crate) size: IDX,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> {
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
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

impl<A: Axis, const K: usize, IDX: Index<T = IDX>> StemNode<A, K, IDX>  {
    pub(crate) fn extend(&mut self, point: &[A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> LeafNode<A, T, K, B, IDX> {
    pub(crate) fn new() -> Self {
        Self {
            min_bound: [A::MAX; K],
            max_bound: [A::MIN; K],
            content_points: [[A::ZERO; K]; B],
            content_items: [T::zero(); B],
            size: IDX::zero(),
        }
    }

    pub(crate) fn extend(&mut self, point: &[A; K]) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> KdTree<A, T, K, B, IDX> where usize: Cast<IDX> {
    #[inline]
    pub fn new() -> Self {
        KdTree::with_capacity(B * 10)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!((capacity.az::<IDX>()) < <IDX as Index>::max());
        let capacity: IDX = capacity.az::<IDX>();
        let mut tree = Self {
            size: T::zero(),
            stems: Vec::with_capacity(capacity.ilog2().az::<usize>()),
            leaves: Vec::with_capacity(capacity.div_ceil(B.az::<IDX>()).az::<usize>()),
            root_index: <IDX as Index>::leaf_offset(),
        };

        tree.leaves.push(LeafNode::new());

        tree
    }

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
