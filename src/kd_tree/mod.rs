mod query_orchestrator;
mod query_stack;
mod traits;

use crate::traits_unified::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;
use aligned_vec::{AVec, CACHELINE_ALIGN};

#[derive(Clone, Debug, PartialEq)]
pub struct KdTree<
    A,              // Axis
    T,              // Content,
    SS,             // StemStrategy
    LS,             // LeafStrategy
    const K: usize, // dimensionality
    const B: usize, // bucket size
> {
    stems: AVec<A>,
    leaves: LS,

    size: usize,
    max_stem_level: i32,
    pub(crate) _phantom: std::marker::PhantomData<(SS, T)>,
}

impl<A, T, SS, LS, const K: usize, const B: usize> Default for KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B> + Default,
    SS: StemStrategy,
{
    fn default() -> Self {
        Self {
            stems: AVec::new(CACHELINE_ALIGN),
            leaves: Default::default(),
            size: 0,
            max_stem_level: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B> + Default,
    SS: StemStrategy,
{
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn max_stem_level(&self) -> i32 {
        self.max_stem_level
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> FromIterator<(T, [A; K])>
    for KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified,
    T: Basics,
    LS: LeafStrategy<A, T, SS, K, B> + Default,
    SS: StemStrategy,
{
    fn from_iter<I: IntoIterator<Item = (T, [A; K])>>(iter: I) -> Self {
        // TODO: Proper impl
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits_unified::{DummyLeafStrategy, FloatMarker};
    use crate::Eytzinger;

    #[test]
    fn test_default() {
        let kd_tree: KdTree<FloatMarker<f32>, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> =
            Default::default();

        assert_eq!(kd_tree.size, 0);
        assert_eq!(kd_tree.max_stem_level, 0);
        assert!(kd_tree.is_empty());
    }

    #[test]
    fn test_from_iterator_empty() {
        let points = vec![[0.0f64; 3]];

        let kd_tree: KdTree<FloatMarker<f64>, u32, Eytzinger<3>, DummyLeafStrategy, 3, 16> =
            points.into_iter().enumerate().collect();

        assert_eq!(kd_tree.size, 0);
    }
}
