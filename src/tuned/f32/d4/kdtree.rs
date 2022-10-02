use std::arch::x86_64::{
    _mm_load_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_store_ps, _mm_storeu_ps,
};
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::tuned::f32::d4::util::{
    distance_to_bounds_simd_f32_4d_squared_euclidean, extend,
};
// The type used to store node indices
pub(crate) type IDX = usize;

// A: Axis, ie points
pub(crate) type A = f32;
// T: Content
pub(crate) type T = usize;
// K: Dimensions
pub(crate) const K: usize = 4;
// B: Bucket size
pub(crate) const B: usize = 32;
pub(crate) type PT = [A; K];

pub(crate) const LEAF_OFFSET: IDX = IDX::MAX.overflowing_shr(1).0;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTree {
    pub leaves: Vec<LeafNode>,
    pub stems: Vec<StemNode>,
    pub(crate) root_index: IDX,
    pub(crate) size: IDX,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
#[repr(align(128))]
pub struct StemNode {
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) min_bound: PT, // 32 * 4 = 128
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    pub(crate) max_bound: PT, // 32 * 4 = 128

    pub(crate) split_val: A, // 32 * 1 =  32

    // TODO: investigate changing usize to u32
    pub(crate) left: IDX,  // 64 * 1 =  64
    pub(crate) right: IDX, // 64 * 1 =  64
} // SIZE   = 416 bits (52 bytes)

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialize_rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
#[repr(align(128))]
pub struct LeafNode {
    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) content_points: [PT; B],

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
    pub(crate) min_bound: PT,

    #[cfg_attr(feature = "serialize", serde(with = "array"))]
    #[cfg_attr(
        feature = "serialize",
        serde(bound(
            serialize = "A: Serialize",
            deserialize = "A: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) max_bound: PT,

    pub(crate) size: IDX,
}

impl StemNode {
    pub(crate) fn extend(&mut self, point: &PT) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
        //unsafe { StemNode::extend_simd_aligned(&mut self.min_bound, &mut self.max_bound, point); }
    }

    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86_64"))]
    #[allow(clippy::missing_safety_doc)]
    #[target_feature(enable = "sse")]
    unsafe fn extend_simd_aligned(min_bound: &mut PT, max_bound: &mut PT, point: &PT) {
        debug_assert!(min_bound.as_ptr() as usize % 16 == 0);
        debug_assert!(max_bound.as_ptr() as usize % 16 == 0);
        debug_assert!(point.as_ptr() as usize % 16 == 0);

        let pt_mm = _mm_load_ps(point.as_ptr());
        let mut max_mm = _mm_load_ps(max_bound.as_ptr());
        let mut min_mm = _mm_load_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_store_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_store_ps(max_bound.as_mut_ptr(), max_mm);
    }

    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86_64"))]
    #[allow(clippy::missing_safety_doc)]
    #[target_feature(enable = "sse")]
    unsafe fn extend_simd_unaligned(min_bound: &mut PT, max_bound: &mut PT, point: &PT) {
        let pt_mm = _mm_loadu_ps(point.as_ptr());
        let mut max_mm = _mm_loadu_ps(max_bound.as_ptr());
        let mut min_mm = _mm_loadu_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_storeu_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_storeu_ps(max_bound.as_mut_ptr(), max_mm);
    }
}

impl LeafNode {
    pub(crate) fn new() -> Self {
        Self {
            min_bound: [A::INFINITY; K],
            max_bound: [A::NEG_INFINITY; K],
            content_points: [[0.0; K]; B],
            content_items: [0; B],
            size: 0,
        }
    }

    pub(crate) fn extend(&mut self, point: &PT) {
        extend(&mut self.min_bound, &mut self.max_bound, point);
        // unsafe { LeafNode::extend_simd_aligned(&mut self.min_bound, &mut self.max_bound, point); }
    }

    pub(crate) fn extend_with_result(&mut self, point: &PT) -> (PT, PT) {
        self.extend(point);
        return (self.min_bound, self.max_bound);
    }

    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86_64"))]
    #[allow(clippy::missing_safety_doc)]
    #[target_feature(enable = "sse")]
    unsafe fn extend_simd_aligned(min_bound: &mut PT, max_bound: &mut PT, point: &PT) {
        debug_assert!(min_bound.as_ptr() as usize % 16 == 0);
        debug_assert!(max_bound.as_ptr() as usize % 16 == 0);
        debug_assert!(point.as_ptr() as usize % 16 == 0);

        let pt_mm = _mm_load_ps(point.as_ptr());
        let mut max_mm = _mm_load_ps(max_bound.as_ptr());
        let mut min_mm = _mm_load_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_store_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_store_ps(max_bound.as_mut_ptr(), max_mm);
    }

    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86_64"))]
    #[allow(clippy::missing_safety_doc)]
    #[target_feature(enable = "sse")]
    unsafe fn extend_simd_unaligned(min_bound: &mut PT, max_bound: &mut PT, point: &PT) {
        let pt_mm = _mm_loadu_ps(point.as_ptr());
        let mut max_mm = _mm_loadu_ps(max_bound.as_ptr());
        let mut min_mm = _mm_loadu_ps(min_bound.as_ptr());

        min_mm = _mm_min_ps(min_mm, pt_mm);
        max_mm = _mm_max_ps(max_mm, pt_mm);

        _mm_storeu_ps(min_bound.as_mut_ptr(), min_mm);
        _mm_storeu_ps(max_bound.as_mut_ptr(), max_mm);
    }

    /*pub(crate) fn calc_bounds(&mut self) {
        self.bounds = [(A::infinity(), A::neg_infinity()); K];
        for idx in 0..self.size {
            self.min_bound.iter_mut().enumerate().for_each(|(dim, bound)| {
                let curr_point_val = *unsafe { self.content.get_unchecked(idx).point.get_unchecked(dim) };
                if curr_point_val < *bound {
                    *bound = curr_point_val;
                }
            });

            self.max_bound.iter_mut().enumerate().for_each(|(dim, bound)| {
                let curr_point_val = *unsafe { self.content.get_unchecked(idx).point.get_unchecked(dim) };
                if curr_point_val > *bound {
                    *bound = curr_point_val;
                }
            });
        }
    }*/
}

impl KdTree {
    #[inline]
    pub fn new() -> Self {
        KdTree::with_capacity((B * 10) as IDX)
    }

    #[inline]
    pub fn with_capacity(capacity: IDX) -> Self {
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
    pub fn size(&self) -> IDX {
        self.size as IDX
    }

    pub(crate) fn is_stem_index(x: IDX) -> bool {
        x < LEAF_OFFSET
    }

    pub(crate) fn child_dist_to_bounds<F>(
        &self,
        query: &PT,
        child_node_idx: IDX,
        _distance_fn: &F,
    ) -> A
    where
        F: Fn(&PT, &PT) -> A,
    {
        if KdTree::is_stem_index(child_node_idx) {
            distance_to_bounds_simd_f32_4d_squared_euclidean(
                query,
                &self.stems[child_node_idx as usize].min_bound,
                &self.stems[child_node_idx as usize].max_bound,
                //distance_fn
            )
        } else {
            distance_to_bounds_simd_f32_4d_squared_euclidean(
                query,
                &self.leaves[(child_node_idx - LEAF_OFFSET) as usize].min_bound,
                &self.leaves[(child_node_idx - LEAF_OFFSET) as usize].max_bound,
                //distance_fn
            )
        }
    }
}

impl Default for KdTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    // use aligned::Aligned;
    use crate::tuned::f32::d4::kdtree::KdTree;

    #[cfg(feature = "serialize")]
    use crate::tuned::f32::d4::kdtree::{PT, T};

    #[test]
    fn it_can_be_constructed_with_new() {
        let tree: KdTree = KdTree::new();

        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn it_can_be_constructed_with_a_defined_capacity() {
        let tree: KdTree = KdTree::with_capacity(10);

        assert_eq!(tree.size(), 0);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn can_serde() {
        let mut tree: KdTree = KdTree::new();

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
    }
}
