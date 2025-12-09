use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, MutableLeafStrategy};
use crate::StemStrategy;
use aligned_vec::AVec;

pub struct VecOfArrays<A, T, const K: usize, const B: usize> {
    leaves: Vec<LeafNode<A, T, K, B>>,
    size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A, T, const K: usize, const B: usize> {
    pub content_points: [[A; B]; K],
    pub content_items: [T; B],
    pub size: usize,
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics,
    SS: StemStrategy,
{
    type Num = AX;

    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaves: Vec::with_capacity(capacity / B + 1),
            size: 0,
        }
    }

    fn bulk_build_from_slice(
        &mut self,
        _source: &[[Self::Num; K]],
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: SS,
    ) -> i32 {
        todo!()
    }

    fn finalize(
        &mut self,
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: &mut SS,
        _max_stem_level: i32,
    ) {
        todo!()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        todo!()
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let leaf = &self.leaves[leaf_idx];

        let points: [&[AX]; K] =
            array_init::array_init(|i| &leaf.content_points[i].as_slice()[..leaf.size]);
        let leaf_items_view = &leaf.content_items[..leaf.size];

        LeafView::new(points, leaf_items_view)
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let leaf_len = leaf_items.len();
        debug_assert!(leaf_len <= B);

        // Sanity: all dims should have the same length
        debug_assert!(leaf_points.iter().all(|p| p.len() == leaf_len));

        // Initialize fixed-size storage with defaults
        let mut content_points: [[AX; B]; K] = array_init::array_init(|_| [AX::zero(); B]);
        let mut content_items: [T; B] = [T::default(); B];

        // Copy the actual data into the first `leaf_len` slots
        for i in 0..leaf_len {
            for dim in 0..K {
                content_points[dim][i] = leaf_points[dim][i];
            }
            content_items[i] = leaf_items[i];
        }

        self.leaves.push(LeafNode {
            content_points,
            content_items,
            size: leaf_len,
        });
        self.size += leaf_len;
    }
}

impl<AX, T, const K: usize, const B: usize> VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified,
    T: Copy + PartialEq,
{
    fn should_remove(
        leaf: &LeafNode<AX, T, K, B>,
        point: &[AX; K],
        item: T,
        leaf_idx: usize,
    ) -> bool {
        for dim in 0..K {
            if leaf.content_points[dim][leaf_idx] != point[dim] {
                return false;
            }
        }

        if leaf.content_items[leaf_idx] != item {
            return false;
        }

        true
    }
}

impl<AX, T, SS, const K: usize, const B: usize> MutableLeafStrategy<AX, T, SS, K, B>
    for VecOfArrays<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics + PartialEq,
    SS: StemStrategy,
{
    fn add_to_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let idx = leaf.size;
        debug_assert!(leaf.size < B, "leaf is full (max capacity reached)");

        for dim in 0..K {
            leaf.content_points[dim][idx] = unsafe { *point.get_unchecked(dim) };
        }
        leaf.content_items[idx] = item;

        leaf.size += 1;
    }

    fn is_leaf_full(&self, leaf_idx: usize) -> bool {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");

        let leaf = unsafe { self.leaves.get_unchecked(leaf_idx) };

        leaf.size >= B
    }

    fn remove_from_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T) {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let mut new_idx = 0;
        for curr_idx in 0..leaf.size {
            // skip items that need removal
            if Self::should_remove(leaf, point, item, curr_idx) {
                continue;
            }

            // if no items have yet needed removal, no action yet needs to be taken
            if new_idx == curr_idx {
                new_idx += 1;
                continue;
            }

            // curr item needs to be kept but needs copying down to lower position in leaf
            for dim in 0..K {
                leaf.content_points[dim][new_idx] = leaf.content_points[dim][curr_idx];
            }
            leaf.content_items[new_idx] = leaf.content_items[curr_idx];
            new_idx += 1;
        }
        self.size -= leaf.size - new_idx;
        leaf.size = new_idx;
    }

    fn split_leaf(&mut self, leaf_idx: usize) -> usize {
        debug_assert!(leaf_idx < self.leaves.len(), "leaf_idx out of bounds");
        let leaf = unsafe { self.leaves.get_unchecked_mut(leaf_idx) };

        let mid = leaf.size / 2;
        leaf.size = mid;

        mid
    }
}

#[cfg(test)]
mod test {
    use fixed::{types::extra::U8, FixedU16};
    use rand::Rng;

    use crate::kd_tree::leaf_strategies::vec_of_arrays::VecOfArrays;
    use crate::traits_unified_2::LeafStrategy;
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn create_single_leaf_vec_of_arrays_float_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view = <VecOfArrays<f32, u32, 3, 32> as LeafStrategy<
            f32,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::leaf_view(&tree.leaves, 0);

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_vec_of_arrays_fixed_point_kd_tree() {
        let points: Vec<[FixedU16<U8>; 3]> = vec![[1.into(), 2.into(), 3.into()]];
        let tree: kd_tree::KdTree<
            FixedU16<U8>,
            u32,
            Eytzinger<3>,
            VecOfArrays<FixedU16<U8>, u32, 3, 32>,
            3,
            32,
        > = kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view = <VecOfArrays<FixedU16<U8>, u32, 3, 32> as LeafStrategy<
            FixedU16<U8>,
            u32,
            Eytzinger<3>,
            3,
            32,
        >>::leaf_view(&tree.leaves, 0);
        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_vec_of_arrays_float_no_items_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];

        let tree: kd_tree::KdTree<f32, (), Eytzinger<3>, VecOfArrays<f32, (), 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice_no_items(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <VecOfArrays<f32, (), 3, 32> as LeafStrategy<f32, (), Eytzinger<3>, 3, 32>>::leaf_view(
                &tree.leaves,
                0,
            );

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![()]);
    }

    #[test]
    fn create_multiple_leaf_vec_of_arrays_float_kd_tree() {
        // create 2^16 random 3d points in the unit cube
        let mut rng = rand::thread_rng();
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let z = rng.gen_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        // perform a nearest_one query
        let query_point = [0.5, 0.5, 0.5];

        let nearest = tree.nearest_one(&query_point);
    }
}
