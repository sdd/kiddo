use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;
use aligned_vec::AVec;

/// A leaf storage strategy using flat vectors for coordinates.
///
/// Stores coordinates as K separate vectors (one per dimension) and items in a separate vector.
pub struct FlatVec<A, T, const K: usize, const B: usize> {
    leaf_points: [Vec<A>; K],
    leaf_items: Vec<T>,
    leaf_extents: Vec<(u32, u32)>,
    size: usize,
}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B>
    for FlatVec<AX, T, K, B>
where
    AX: AxisUnified<Coord = AX>,
    T: Basics,
    SS: StemStrategy,
{
    type Num = AX;

    fn new_with_capacity(capacity: usize) -> Self {
        Self {
            leaf_points: array_init::array_init(|_| Vec::with_capacity(capacity)),
            leaf_items: Vec::with_capacity(capacity),
            leaf_extents: Vec::with_capacity(capacity),
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
        self.leaf_extents.len()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        todo!()
    }

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        let (start, end) = self.leaf_extents[leaf_idx];

        let leaf_points_view =
            array_init::array_init(|i| &self.leaf_points[i][start as usize..end as usize]);

        let leaf_items_view = &self.leaf_items[start as usize..end as usize];

        LeafView::new(leaf_points_view, leaf_items_view)
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let chunk_length = leaf_items.len();

        debug_assert!(leaf_points[0].len() == chunk_length);
        for d in leaf_points.iter() {
            debug_assert!(d.len() == chunk_length);
        }

        self.leaf_extents.push((
            self.leaf_items.len() as u32,
            (self.leaf_items.len() + chunk_length) as u32,
        ));

        for dim in 0..K {
            self.leaf_points[dim].extend_from_slice(&leaf_points[dim][..chunk_length]);
        }
        self.leaf_items
            .extend_from_slice(&leaf_items[..chunk_length]);
    }
}

#[cfg(test)]
mod test {
    use fixed::{types::extra::U8, FixedU16};
    use rand::Rng;
    use std::num::NonZeroUsize;

    use crate::kd_tree::leaf_strategies::flat_vec::FlatVec;
    use crate::traits_unified_2::{LeafStrategy, SquaredEuclidean};
    use crate::{kd_tree, Eytzinger};

    #[test]
    fn create_single_leaf_flat_vec_float_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <FlatVec<f32, u32, 3, 32> as LeafStrategy<f32, u32, Eytzinger<3>, 3, 32>>::leaf_view(
                &tree.leaves,
                0,
            );

        let (leaf_points, leaf_items) = leaf_view.into_parts();
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[1][0], points[0][1]);
        assert_eq!(leaf_points[2][0], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }

    #[test]
    fn create_single_leaf_flat_vec_float_no_items_kd_tree() {
        let points: Vec<[f32; 3]> = vec![[1.0f32, 2.0f32, 3.0f32]];
        let tree: kd_tree::KdTree<f32, (), Eytzinger<3>, FlatVec<f32, (), 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice_no_items(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view =
            <FlatVec<f32, (), 3, 32> as LeafStrategy<f32, (), Eytzinger<3>, 3, 32>>::leaf_view(
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
    fn create_single_leaf_flat_vec_fixed_point_kd_tree() {
        let points: Vec<[FixedU16<U8>; 3]> = vec![[1.into(), 2.into(), 3.into()]];
        let tree: kd_tree::KdTree<
            FixedU16<U8>,
            u32,
            Eytzinger<3>,
            FlatVec<FixedU16<U8>, u32, 3, 32>,
            3,
            32,
        > = kd_tree::KdTree::new_from_slice(&points);

        assert_eq!(tree.size(), 1);

        let leaf_view = <FlatVec<FixedU16<U8>, u32, 3, 32> as LeafStrategy<
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
    fn create_multiple_leaf_flat_vec_float_kd_tree() {
        // create 2^16 random 3d points in the unit cube
        let mut rng = rand::rng();
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: kd_tree::KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            kd_tree::KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        // perform a best_n_within query
        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();
        let results = tree.best_n_within::<SquaredEuclidean<f32>>(&query_point, radius, max_qty);
        assert_eq!(results.len(), 10);
    }
}
