use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        let req_ctx = GetLeafIdxReqCtx { query };
        let mut leaf_idx = 0;
        self.straight_query(req_ctx, |_, l| {
            leaf_idx = l;
            true
        });

        leaf_idx
    }
}

struct GetLeafIdxReqCtx<'a, A, const K: usize> {
    query: &'a [A; K],
}

impl<A, const K: usize> QueryContext<A, K> for GetLeafIdxReqCtx<'_, A, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits_unified_2::DummyLeafStrategy;
    use crate::Eytzinger;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree.get_leaf_idx(&query);

        assert_eq!(result, 3);
    }
}
