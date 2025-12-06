use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, LeafView};
use crate::StemStrategy;
use aligned_vec::AVec;

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
        stem_strategy: SS,
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

        (leaf_points_view, leaf_items_view)
    }

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]) {
        let chunk_length = leaf_items.len();

        debug_assert!(leaf_points[0].len() == chunk_length);
        for d in leaf_points.iter() {
            debug_assert!(d.len() == chunk_length);
        }

        self.leaf_extents.push((
            leaf_items.len() as u32,
            (leaf_items.len() + chunk_length) as u32,
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
    use crate::kd_tree::leaf_strategies::flat_vec::FlatVec;
    use crate::traits_unified_2::LeafStrategy;
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
        let (leaf_points, leaf_items) = leaf_view;
        assert_eq!(leaf_points[0][0], points[0][0]);
        assert_eq!(leaf_points[0][1], points[0][1]);
        assert_eq!(leaf_points[0][2], points[0][2]);
        assert_eq!(leaf_items, vec![0]);
    }
}
