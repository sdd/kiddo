use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::StemStrategy;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds an approximate nearest point to the query point.
    ///
    /// This is faster than `nearest_one` but may not return the true nearest neighbor.
    /// It searches only the leaf that the query point falls into.
    pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>,
    {
        let req_ctx = ApproxNearestOneReqCtx::<A, D::Output, K> {
            query,
            _phantom: std::marker::PhantomData,
        };

        let mut best_dist = D::Output::max_value();
        let mut best_item = T::default();

        self.straight_query(req_ctx, |leaf| {
            leaf.nearest_one::<D>(query, &mut best_dist, &mut best_item);
        });

        (best_dist, best_item)
    }
}

struct ApproxNearestOneReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    _phantom: std::marker::PhantomData<O>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ApproxNearestOneReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        panic!("approx_nearest_one should not be called with max_dist")
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn approx_nearest_one_flat_vec_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0014114721, 19074));
    }

    #[test]
    fn approx_nearest_one_vec_of_arrays_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0014114721, 19074));
    }

    #[test]
    fn approx_nearest_one_vec_of_arrays_mutated_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx as u32)
        }

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0014114721, 19074));
    }
}
