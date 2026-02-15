use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics, BucketLimitType, Immutable, LeafStrategy};
use crate::StemStrategy;

/// A dummy leaf strategy used for testing.
///
/// This strategy provides no-op implementations and is not meant for production use.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct DummyLeafStrategy {}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B> for DummyLeafStrategy
where
    AX: AxisUnified,
    T: Basics,
    SS: StemStrategy,
{
    type Num = ();
    type Mutability = Immutable;

    const BUCKET_LIMIT_TYPE: BucketLimitType = BucketLimitType::Hard;

    fn new_with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    fn new_with_empty_leaf() -> Self {
        Self::default()
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
