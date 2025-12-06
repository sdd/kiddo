use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy, LeafView};
use crate::StemStrategy;
use aligned_vec::AVec;

#[derive(Debug, Default)]
pub struct DummyLeafStrategy {}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B> for DummyLeafStrategy
where
    AX: AxisUnified,
    T: Basics,
    SS: StemStrategy,
{
    type Num = ();

    fn new_with_capacity(_capacity: usize) -> Self {
        unimplemented!()
    }

    fn bulk_build_from_slice(
        &mut self,
        _source: &[[Self::Num; K]],
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: SS,
    ) -> i32 {
        unimplemented!()
    }

    fn finalize(
        &mut self,
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: &mut SS,
        _max_stem_level: i32,
    ) {
        unimplemented!()
    }

    fn size(&self) -> usize {
        unimplemented!()
    }

    fn leaf_count(&self) -> usize {
        unimplemented!()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        unimplemented!()
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K, B> {
        unimplemented!()
    }

    fn append_leaf(&mut self, _leaf_points: &[&[AX]; K], _leaf_items: &[T]) {
        // NOOP
    }
}
