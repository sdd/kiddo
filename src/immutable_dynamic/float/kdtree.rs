//! Immutable Floating point k-d tree. Offers less memory utilisation, smaller size
//! when serialized, and faster more consistent query performance. This comes at the
//! expense of not being able to modify the contents of the tree after its initial
//! construction, and longer construction times - perhaps prohibitively so.
//! As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
//! values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled

use aligned_vec::{avec,AVec};
use array_init::array_init;
use az::{Az, Cast};
use ordered_float::OrderedFloat;
use std::cmp::PartialEq;
use std::fmt::Debug;

#[cfg(feature = "tracing")]
use tracing::{event, span, Level};

pub use crate::float::kdtree::Axis;
use crate::float_leaf_simd::leaf_node::BestFromDists;
use crate::types::Content;

use crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Immutable Dynamic floating point k-d tree
///
/// Offers less memory utilisation, smaller size vs non-immutable tree
/// when serialized, and faster more consistent query performance. This comes at the
/// expense of not being able to modify the contents of the tree after its initial
/// construction, and longer construction times.
///
/// Compared to non-dynamic ImmutableDynamicKdTree, this can handle data like point clouds
/// that may have many occurrences of multiple points have the exact same value on a given axis.
/// This comes at the expense of slower performance. Memory usage should still be very efficient,
/// more so than the standard and non-dynamic immutable tree types.
///
/// As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
/// values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled
///
/// A convenient type alias exists for ImmutableDynamicKdTree with some sensible defaults set: [`kiddo::ImmutableDynamicKdTree`](`crate::ImmutableDynamicKdTree`).
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ImmutableDynamicKdTree<
    A: Copy + Default,
    T: Copy + Default,
    const K: usize,
    const B: usize,
> {
    pub(crate) stems: AVec<A>,
    pub(crate) leaf_points: [Vec<A>; K],
    pub(crate) leaf_items: Vec<T>,
    pub(crate) leaf_extents: Vec<(u32,u32)>,
    pub(crate) max_stem_level: usize,
}

impl<A: Axis, T: Content, const K: usize, const B: usize> From<&[[A; K]]>
    for ImmutableDynamicKdTree<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableDynamicKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableDynamicKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable_dynamic::float::kdtree::ImmutableDynamicKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableDynamicKdTree<f64, u32, 3, 32> = (&*points).into();
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    fn from(slice: &[[A; K]]) -> Self {
        ImmutableDynamicKdTree::new_from_slice(slice)
    }
}

// prevent clippy complaining that the feature unreliable_select_nth_unstable
// is not defined (I don't want to explicitly define it as if I do then
// passing --all-features in CI will enable it, which I don't want to do
#[allow(unexpected_cfgs)]
impl<A, T, const K: usize, const B: usize> ImmutableDynamicKdTree<A, T, K, B>
where
    A: Axis + BestFromDists<T, B>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableDynamicKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableDynamicKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable_dynamic::float::kdtree::ImmutableDynamicKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableDynamicKdTree<f64, u32, 3, 32> = ImmutableDynamicKdTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn new_from_slice(source: &[[A; K]]) -> Self
    where
        usize: Cast<T>,
    {
        let item_count = source.len();
        let leaf_node_count = item_count.div_ceil(B);
        let max_stem_level: usize = (leaf_node_count.next_power_of_two().ilog2() - 1) as usize;
        let stem_node_count = leaf_node_count.next_power_of_two() - 1;

        // TODO: this is wrong for most situations needing > 7 nodes
        // let stem_node_count = stem_node_count + stem_node_count.div_floor(7);
        let stem_node_count = stem_node_count * 5;
        // TODO: just trim the stems afterwards by traversing right-child non-inf nodes
        //       till we hit max level to get the max used stem

        let mut stems = avec![A::infinity(); stem_node_count];
        let mut leaf_points: [Vec<A>; K] = array_init(|_| Vec::with_capacity(item_count));
        let mut leaf_items: Vec<T> = Vec::with_capacity(item_count);
        let mut leaf_extents: Vec<(u32,u32)> = Vec::with_capacity(item_count.div_ceil(B));

        let mut sort_index = Vec::from_iter(0..item_count);

        Self::populate_recursive(
            &mut stems,
            0,
            source,
            &mut sort_index,
            0,
            0,
            max_stem_level,
            leaf_node_count * B,
            &mut leaf_points,
            &mut leaf_items,
            &mut leaf_extents,
        );

        Self {
            stems,
            leaf_points,
            leaf_items,
            leaf_extents,
            max_stem_level,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_recursive(
        stems: &mut AVec<A>,
        dim: usize,
        source: &[[A; K]],
        sort_index: &mut [usize],
        stem_index: usize,
        mut level: usize,
        max_stem_level: usize,
        capacity: usize,
        leaf_points: &mut [Vec<A>; K],
        leaf_items: &mut Vec<T>,
        leaf_extents: &mut Vec<(u32,u32)>,
    ) {
        #[cfg(feature = "tracing")]
        let span = span!(Level::TRACE, "opt", idx = stem_index);
        #[cfg(feature = "tracing")]
        let _enter = span.enter();
        let chunk_length = sort_index.len();

        if level > max_stem_level {
            // Write leaf and terminate recursion
            // println!("Writing leaf #{:?}", leaf_extents.len());

            leaf_extents.push((leaf_items.len() as u32, (leaf_items.len() + chunk_length) as u32));

            (0..chunk_length).for_each(|i| {
                (0..K).for_each(|dim| leaf_points[dim].push(source[sort_index[i]][dim]));
                leaf_items.push(sort_index[i].az::<T>())
            });

            return;
        }

        // println!("Handling stem #{:?}", stem_index);
        let levels_below = max_stem_level - level;
        let left_capacity = (2usize.pow(levels_below as u32) * B).min(capacity);
        let right_capacity = capacity.saturating_sub(left_capacity);

        let mut pivot = Self::calc_pivot(chunk_length, stem_index, right_capacity);

        // only bother with this if we are putting at least one item in the right hand child
        if pivot < chunk_length {
            pivot = Self::update_pivot(source, sort_index, dim, pivot);

            // if we end up with a pivot of 0, something has gone wrong,
            // unless we only had a slice of len 1 anyway
            debug_assert!(pivot > 0 || chunk_length == 1);
            debug_assert!(
                stems[stem_index].is_infinite(),
                "Wrote to stem #{:?} for a second time",
                stem_index
            );

            stems[stem_index] = source[sort_index[pivot]][dim];
        }

        let left_child_idx = modified_van_emde_boas_get_child_idx_v2(stem_index, false, level);
        let right_child_idx = modified_van_emde_boas_get_child_idx_v2(stem_index, true, level);

        let (lower_sort_index, upper_sort_index) = sort_index.split_at_mut(pivot);

        level += 1;
        let next_dim = (dim + 1) % K;

        Self::populate_recursive(
            stems,
            next_dim,
            source,
            lower_sort_index,
            left_child_idx,
            level,
            max_stem_level,
            left_capacity,
            leaf_points,
            leaf_items,
            leaf_extents,
        );

        Self::populate_recursive(
            stems,
            next_dim,
            source,
            upper_sort_index,
            right_child_idx,
            level,
            max_stem_level,
            right_capacity,
            leaf_points,
            leaf_items,
            leaf_extents,
        );
    }

    #[cfg(not(feature = "unreliable_select_nth_unstable"))]
    #[inline]
    fn update_pivot(
        source: &[[A; K]],
        sort_index: &mut [usize],
        dim: usize,
        mut pivot: usize,
    ) -> usize {
        // TODO: this block might be faster by using a quickselect with a fat partition?
        //       we could then run that quickselect and subtract (fat partition length - 1)
        //       from the pivot, avoiding the need for the while loop.

        // ensure the item whose index = pivot is in its correctly sorted position, and any
        // items that are equal to it are adjacent, according to our assumptions about the
        // behaviour of `select_nth_unstable_by` (See examples/check_select_nth_unstable.rs)
        sort_index.select_nth_unstable_by_key(pivot, |&i| OrderedFloat(source[i][dim]));

        if pivot == 0 {
            event!(
                Level::WARN,
                pivot,
                chunk_len = sort_index.len(),
                pivot0_val = ?source[sort_index[pivot]][dim],
                pivot1_val = ?source[sort_index[pivot + 1]][dim],
                "Pivot already at 0. Update-pivot can't move it left."
            );
            return pivot;
        }

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 1 {
            pivot -= 1;
            #[cfg(feature = "tracing")]
            event!(
                Level::INFO,
                pivot,
                chunk_len = sort_index.len(),
                pivotN1_val = ?source[sort_index[pivot - 1]][dim],
                pivot0_val = ?source[sort_index[pivot]][dim],
                pivot1_val = ?source[sort_index[pivot + 1]][dim],
                "pivot shifted"
            );
        }

        pivot
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use kiddo::immutable_dynamic::float::kdtree::ImmutableDynamicKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableDynamicKdTree<f64, u32, 3, 32> = ImmutableDynamicKdTree::new_from_slice(&points);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        self.leaf_items.len()
    }

    /// Returns the theoretical max capacity of this tree
    #[inline]
    pub fn capacity(&self) -> usize {
        self.size()
    }

    fn calc_pivot(chunk_length: usize, _stem_index: usize, _right_capacity: usize) -> usize {
        chunk_length >> 1
    }
}

#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > ArchivedImmutableDynamicKdTree<A, T, K, B>
{
    /// Returns the current number of elements stored in the tree
    #[inline]
    pub fn size(&self) -> usize {
        self.leaf_items.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::immutable_dynamic::float::kdtree::ImmutableDynamicKdTree;
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};
    use rayon::prelude::IntoParallelRefIterator;
    use std::panic;

    #[test]
    fn can_construct_optimized_tree_with_straddled_split() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
        ];

        let tree: ImmutableDynamicKdTree<f32, usize, 2, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);

        // assert_eq!(tree.leaf_extents[0].iter().count(), 3);
        // assert_eq!(tree.leaf_extents[1].iter().count(), 5);
        // assert_eq!(tree.leaf_extents[2].iter().count(), 4);
        // assert_eq!(tree.leaf_extents[3].iter().count(), 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_2() {
        let content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];

        let tree: ImmutableDynamicKdTree<f32, usize, 2, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_with_straddled_split_3() {
        use rand::seq::SliceRandom;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(23);

        let mut content_to_add = vec![
            [1.0, 101.0],
            [2.0, 102.0],
            [3.0, 103.0],
            [4.0, 104.0],
            [4.0, 104.0],
            [5.0, 105.0],
            [6.0, 106.0],
            [7.0, 107.0],
            [8.0, 108.0],
            [9.0, 109.0],
            [10.0, 110.0],
            [11.0, 111.0],
            [12.0, 112.0],
            [13.0, 113.0],
            [14.0, 114.0],
            [15.0, 115.0],
            [16.0, 116.0],
            [17.0, 117.0],
            [18.0, 118.0],
        ];
        content_to_add.shuffle(&mut rng);

        let _tree: ImmutableDynamicKdTree<f32, usize, 2, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);

        // assert_eq!(tree.leaves[0].size, 3);
        // assert_eq!(tree.leaves[1].size, 4);
        // assert_eq!(tree.leaves[2].size, 4);
        // assert_eq!(tree.leaves[3].size, 4);
        // assert_eq!(tree.leaves[4].size, 4);
    }

    #[test]
    fn can_construct_optimized_tree_with_multiple_dupes() {
        use rand::seq::SliceRandom;

        for seed in 0..1_000 {
            //_000 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

            let mut content_to_add = vec![
                [4.0, 104.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [4.0, 104.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [4.0, 104.0],
                [4.0, 104.0],
                [17.0, 117.0],
                [18.0, 118.0],
            ];
            content_to_add.shuffle(&mut rng);

            let _tree: ImmutableDynamicKdTree<f32, usize, 2, 8> =
                ImmutableDynamicKdTree::new_from_slice(&content_to_add);
        }
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_0() {
        let tree_size = 18;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);

        println!("tree: {:?}", tree);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_1() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_2() {
        let tree_size = 155;
        let seed = 480;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_3() {
        let tree_size = 26; // also 32
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_4() {
        let tree_size = 21;
        let seed = 131851;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_5() {
        let tree_size = 32;
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_6() {
        let tree_size = 56;
        let seed = 450533;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_7() {
        let tree_size = 18;
        let seed = 992063;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_8() {
        let tree_size = 19;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_9() {
        let tree_size = 20;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_10() {
        let tree_size = 36;
        let seed = 375096;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_11() {
        let tree_size = 10000;
        let seed = 257281;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_many_dupes() {
        let tree_size = 8;
        let seed = 0;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let mut duped: Vec<[f32; 4]> = Vec::with_capacity(content_to_add.len() * 10);
        for item in content_to_add {
            for _ in 0..6 {
                duped.push(item);
            }
        }

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 8> =
            ImmutableDynamicKdTree::new_from_slice(&duped);
    }

    #[ignore]
    #[test]
    fn can_construct_optimized_tree_multi_rand_increasing_size() {
        use rayon::iter::ParallelIterator;

        #[allow(dead_code)]
        #[derive(Debug)]
        struct Failure {
            tree_size: i32,
            seed: u64,
        }

        let failures: Vec<Failure> = Vec::new();

        for tree_size in 16..100 {
            (0..1_000_000)
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(|&seed| {
                    let result = panic::catch_unwind(|| {
                        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                        let content_to_add: Vec<[f32; 4]> =
                            (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

                        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
                            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
                    });

                    if result.is_err() {
                        //failures.push(Failure { tree_size, seed });
                        println!("Failed on tree size {}, seed #{}", tree_size, seed);
                    }
                });
        }

        println!("{:?}", &failures);
        assert!(failures.is_empty());
    }

    #[test]
    fn can_construct_optimized_tree_medium_rand() {
        use itertools::Itertools;

        const TREE_SIZE: usize = 2usize.pow(19); // ~ 500k

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let num_uniq = content_to_add
            .iter()
            .flatten()
            .map(|&x| OrderedFloat(x))
            .unique()
            .count();

        println!("dupes: {:?}", TREE_SIZE * 4 - num_uniq);

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 4> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[ignore]
    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableDynamicKdTree<f32, usize, 4, 32> =
            ImmutableDynamicKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_tree_claytonwramsey_case() {
        let points = [
            [0.29224798, 0.9763673, -0.3175672],
            [0.67914444, 0.9027409, 0.024138344],
            [0.6523841, 0.9036499, 0.058751766],
            [0.2740701, 0.9650571, -0.33],
            [-0.02133016, 0.7669279, -0.26754907],
            [0.25805938, 1.001667, -0.19],
            [0.63938284, 0.860512, 0.0107941795],
            [-0.024343323, 0.72316134, -0.3087369],
            [0.28637683, 0.96052605, -0.2653344],
            [-0.015163747, 0.73591834, -0.22867745],
            [0.5940022, 0.5538876, 0.04227795],
            [-0.027717995, 0.72078663, -0.28554514],
            [-0.07310812, 0.7494081, -0.23818146],
            [0.6659923, 0.87172854, -0.08],
            [0.66515255, 0.84636533, -0.034273863],
            [0.5710293, 0.5819249, -0.06687962],
            [0.68895954, 0.89422065, -0.021492248],
            [0.23974295, 0.99820554, -0.25543308],
            [-0.044163335, 0.71646404, -0.26691338],
            [0.6765435, 0.87821645, 0.06],
            [0.6505354, 0.85000074, 0.007534381],
            [0.27864423, 1.0036206, -0.2321812],
            [-0.18781066, 0.7993474, 0.73455876],
            [-0.030041732, 0.7589699, -0.33],
            [-0.16606936, 0.7862105, 0.81],
            [0.26586565, 1.0082403, -0.2965155],
            [-0.016885525, 0.73214334, -0.20020396],
            [-0.014521084, 0.73782456, -0.30040723],
            [0.5984133, 0.5876088, -0.08],
            [0.67026573, 0.8468562, 0.030995809],
            [0.6943986, 0.88156444, 0.033436965],
            [-0.18674837, 0.785625, 0.67],
            [0.69352907, 0.885177, -0.05081772],
            [-0.07323426, 0.74754643, -0.22428818],
            [0.6348539, 0.87661994, -0.011788521],
            [0.66116154, 0.8465918, -0.058175772],
            [0.613001, 0.58581233, 0.06],
            [0.690586, 0.8609429, 0.0040903892],
            [0.6922138, 0.88866735, -0.033763546],
            [-0.19691803, 0.77833843, 0.75353056],
            [0.66047364, 0.86635274, 0.06],
            [-0.13778007, 0.78493524, 0.8024779],
            [0.25623387, 1.0078267, -0.26043046],
            [-0.19601217, 0.77047926, 0.6762747],
            [0.6771497, 0.8760619, 0.06],
            [-0.061780274, 0.77004665, -0.2525345],
            [-0.18836515, 0.79879534, 0.7045418],
            [-0.15025152, 0.7528725, 0.8075481],
            [-0.040078525, 0.71661884, -0.26288617],
            [-0.03125439, 0.7189548, -0.24654058],
            [0.6652675, 0.8463667, 0.053869896],
            [-0.02844002, 0.7569768, -0.33],
            [-0.013920627, 0.7401625, -0.26471153],
            [0.60257894, 0.55309933, 0.046852764],
            [0.29219308, 0.9812116, -0.19597858],
            [0.28911182, 0.9649517, -0.24913748],
            [-0.07281278, 0.7515782, -0.27513155],
            [-0.13923754, 0.78936976, 0.8005492],
            [0.62335277, 0.56304044, 0.009596429],
            [-0.14526843, 0.79857683, 0.69145817],
            [0.28210846, 0.9559021, -0.29579282],
            [-0.19371477, 0.7913114, 0.69353455],
            [-0.046852536, 0.77623373, -0.30445236],
            [0.28982612, 0.990427, -0.24529156],
            [0.636859, 0.8655774, -0.0018347164],
            [-0.021187682, 0.7261263, -0.3166377],
            [-0.15338133, 0.7668287, 0.81],
            [0.69484174, 0.8771785, -0.007962311],
            [-0.06919295, 0.7313776, -0.19203408],
            [-0.17287138, 0.8028957, 0.81],
            [-0.040346064, 0.776309, -0.26601598],
            [0.60512835, 0.5978099, 0.06],
            [-0.16278483, 0.7481007, 0.7795725],
            [0.69397146, 0.86914545, 0.04912829],
            [-0.014357822, 0.7545122, -0.28273812],
            [-0.13704276, 0.7804961, 0.7320505],
            [-0.05018555, 0.71909815, -0.33],
            [0.25258696, 0.9500735, -0.31038237],
            [-0.140453, 0.7919334, 0.6849478],
            [0.6017555, 0.56755596, -0.08],
            [0.6516504, 0.9033026, -0.04285966],
            [-0.14095415, 0.79283524, 0.79076195],
            [0.28638643, 0.9940055, -0.19],
            [-0.024419375, 0.72309977, -0.22033331],
            [0.25758767, 0.97917306, -0.33],
            [0.28283247, 1.0003425, -0.29746082],
            [0.6560958, 0.8476704, -0.013107247],
            [-0.1927107, 0.79314286, 0.76401347],
            [-0.016432043, 0.7598878, -0.29553702],
            [-0.16678295, 0.7774815, 0.81],
            [-0.19663718, 0.77368546, 0.7684957],
            [-0.14910127, 0.753681, 0.68857074],
            [-0.026120612, 0.75759894, -0.33],
            [0.666949, 0.90629053, -0.05558681],
            [0.23726322, 0.961954, -0.19168153],
            [0.6639612, 0.90635055, -0.059393443],
            [0.2687751, 1.007748, -0.21141726],
            [0.6264445, 0.59896344, 0.016942762],
            [-0.06529051, 0.7668072, -0.20704435],
            [0.28878632, 0.99257696, -0.21497893],
            [0.27467704, 1.0057876, -0.21584071],
            [-0.05297368, 0.7748322, -0.24199645],
            [-0.070086524, 0.7330327, -0.26519936],
            [-0.022016877, 0.72526133, -0.20158654],
            [-0.14319588, 0.7961727, 0.7928342],
            [-0.14104009, 0.76264507, 0.71503407],
            [0.25688088, 1.0079533, -0.2839761],
            [0.23446296, 0.98958445, -0.23201855],
            [0.64889103, 0.8561075, -0.08],
            [0.67663074, 0.8487725, -0.053594552],
            [0.28375882, 0.957465, -0.2904101],
            [-0.02791013, 0.7206713, -0.29259408],
            [-0.038439978, 0.7760615, -0.20030592],
            [-0.061959196, 0.7699051, -0.31189045],
            [-0.046509318, 0.7432705, -0.33],
            [-0.07283273, 0.7414387, -0.2536798],
            [-0.0696171, 0.7607682, -0.32882372],
            [-0.17498404, 0.74891734, 0.6775535],
            [-0.04342684, 0.71645075, -0.3193106],
            [0.6694648, 0.90147305, -0.08],
            [-0.03841278, 0.776057, -0.2904402],
            [0.60229224, 0.6082346, 0.06],
            [0.63623226, 0.8673714, -0.07047073],
            [0.25525373, 0.94929475, -0.22928719],
            [-0.013307185, 0.74823093, -0.25914216],
            [-0.15204464, 0.7789803, 0.67],
            [0.66158736, 0.84654206, -0.016549766],
            [-0.18908727, 0.7575969, 0.7947485],
            [-0.030717596, 0.72964686, -0.19],
            [0.6407536, 0.8942308, 0.0030672126],
            [-0.07323421, 0.747548, -0.27356464],
            [-0.15000835, 0.7530368, 0.6837221],
            [-0.02280628, 0.72449857, -0.29852086],
            [-0.13852423, 0.7874851, 0.7450513],
            [0.26314273, 0.9830219, -0.33],
            [0.24466209, 0.95419806, -0.19090004],
            [0.25458035, 0.9631764, -0.19],
            [-0.16561191, 0.80778533, 0.7786589],
            [-0.15712304, 0.8061683, 0.7941575],
            [-0.1950157, 0.7672888, 0.7814798],
            [0.6688157, 0.8657684, -0.08],
            [0.29217386, 0.9814121, -0.27473232],
            [0.26246512, 0.9711056, -0.33],
            [-0.03280885, 0.740659, -0.19],
            [0.69342744, 0.8672266, -0.009337944],
            [-0.07211808, 0.7382722, -0.25419936],
            [0.6522962, 0.8569991, 0.06],
            [-0.19616698, 0.7711231, 0.7174622],
            [0.29232034, 0.9783973, -0.21790323],
            [-0.017751085, 0.730652, -0.2037139],
            [-0.19391096, 0.7909147, 0.73797905],
            [0.64591116, 0.8530998, -0.04372617],
            [0.6453618, 0.8535581, -0.005502467],
            [0.2612535, 0.9785364, -0.33],
            [0.6943786, 0.87105095, 0.058666393],
            [0.27023894, 1.0073867, -0.27258444],
            [0.6820491, 0.85178155, 0.02813362],
            [-0.061896227, 0.72294545, -0.26750845],
            [0.29229453, 0.9796962, -0.20070004],
            [-0.14229497, 0.7606827, 0.68364507],
            [0.66606385, 0.90633935, -0.059026986],
            [0.28388107, 0.95815027, -0.33],
            [0.25858292, 1.0082169, -0.24233256],
            [-0.01370127, 0.7516096, -0.23526654],
            [0.58405435, 0.6078084, -0.02321302],
            [0.25908658, 1.0082757, -0.23670314],
            [-0.051672924, 0.77524483, -0.25714234],
            [0.24397893, 1.0021906, -0.2303035],
            [-0.13747106, 0.78352416, 0.7645246],
            [0.62125045, 0.6051995, -0.02051663],
            [-0.04264366, 0.7455886, -0.19],
            [0.5808382, 0.6052659, 0.028435156],
            [0.65270334, 0.9037936, 0.0013047489],
            [-0.13694426, 0.77895343, 0.7421566],
            [0.2696836, 0.94936824, -0.316583],
            [-0.024855208, 0.72275484, -0.31240752],
            [-0.13788, 0.7853324, 0.8008346],
            [-0.013781451, 0.74085104, -0.19861262],
            [0.69476223, 0.8740344, -0.023301095],
            [0.28553954, 0.97520775, -0.19],
            [-0.19582705, 0.78584725, 0.7665467],
            [0.25396827, 1.0072645, -0.23788433],
            [0.25354952, 0.9606116, -0.19],
            [0.5885661, 0.5557597, -0.014533235],
            [0.6804821, 0.9019709, -0.059310704],
            [-0.05574975, 0.77372414, -0.3148804],
            [-0.14085324, 0.7926594, 0.7194474],
            [-0.17620997, 0.7988105, 0.81],
            [0.29083416, 0.96912503, -0.21674244],
            [0.6925983, 0.8649539, 0.023039693],
            [0.28517142, 0.96564835, -0.19],
            [-0.06204924, 0.76983297, -0.3256017],
            [-0.027590564, 0.7208642, -0.29954094],
            [-0.056383595, 0.77342474, -0.27509728],
            [0.23344278, 0.9865797, -0.25063255],
            [0.27788487, 1.0020956, -0.19],
            [-0.03230455, 0.7743806, -0.19277528],
            [0.25882086, 1.0082457, -0.32201782],
            [0.23986866, 0.95855284, -0.29807574],
            [-0.14315465, 0.7961193, 0.7088207],
            [0.6634052, 0.8463988, -0.006263792],
            [0.6367079, 0.8867496, -0.035606433],
            [-0.19262113, 0.7932925, 0.73465395],
            [-0.070361264, 0.7335968, -0.26570734],
            [-0.16632818, 0.7796913, 0.81],
            [0.63601094, 0.86810863, -0.0047814306],
            [-0.14577873, 0.7990962, 0.7250501],
            [-0.013257537, 0.7460091, -0.20899436],
            [0.68454766, 0.8537339, -0.026210718],
            [-0.02693921, 0.72127455, -0.21019574],
            [-0.17310655, 0.7973076, 0.81],
            [-0.06845559, 0.7627259, -0.3183744],
            [-0.1664119, 0.8078096, 0.79643416],
            [-0.03777641, 0.7419998, -0.33],
            [-0.16569902, 0.78068906, 0.67],
            [-0.1911147, 0.76007295, 0.7139923],
            [0.6848929, 0.8540391, 0.019340616],
            [-0.041500412, 0.77639896, -0.31096855],
            [0.59875125, 0.55314314, 0.00385536],
            [0.6706438, 0.8800843, 0.06],
            [0.63700885, 0.88753146, 0.0012290822],
            [0.57120425, 0.5796317, -0.061768703],
            [0.2921545, 0.98160124, -0.32111108],
            [-0.04908788, 0.7355092, -0.33],
            [-0.015797654, 0.75853884, -0.32312664],
            [0.5710877, 0.58087164, 0.058441624],
            [0.27745405, 1.0043538, -0.32905272],
            [0.2900878, 0.967094, -0.3065594],
            [-0.03790397, 0.7282038, -0.19],
            [0.57559025, 0.5671226, 0.023510179],
            [-0.042243835, 0.74772173, -0.33],
            [-0.19145454, 0.795082, 0.7436563],
            [0.65927446, 0.89858633, 0.06],
            [0.67694247, 0.90382, 0.037180822],
            [-0.17009991, 0.7479827, 0.67558026],
            [-0.057965703, 0.7725955, -0.24140492],
            [0.2714524, 0.9498742, -0.2696474],
            [0.64330244, 0.85549325, 0.051631734],
            [0.68751734, 0.8567089, 0.030820819],
            [0.63714045, 0.8878541, -0.03201224],
            [-0.027051583, 0.7212021, -0.28615162],
            [0.24515572, 0.9538462, -0.22268322],
            [-0.020121725, 0.7273483, -0.2475043],
            [-0.15810107, 0.8064876, 0.70368385],
            [-0.13985121, 0.7907422, 0.7721606],
            [0.6279307, 0.5698232, 0.017414112],
            [0.23613273, 0.99308634, -0.23573668],
            [-0.16718502, 0.74781513, 0.6883865],
            [0.28194004, 0.9557554, -0.2896209],
            [-0.1929176, 0.7927893, 0.7619351],
            [-0.14917459, 0.753627, 0.7579014],
            [-0.039151255, 0.7167322, -0.27025294],
            [-0.060107302, 0.7216314, -0.29650208],
            [-0.044772007, 0.7764119, -0.20262036],
            [0.2803882, 1.0023996, -0.30964604],
            [-0.030847529, 0.7737646, -0.24624208],
            [-0.1677756, 0.77728736, 0.81],
            [0.63843894, 0.8905874, 0.0029261732],
            [0.2676363, 0.9489253, -0.32097113],
            [0.666439, 0.8777975, 0.06],
            [0.6586371, 0.8822851, -0.08],
            [-0.17900534, 0.7761261, 0.81],
            [0.67531496, 0.84824723, 0.029400958],
            [0.2826386, 1.0005225, -0.30394673],
            [0.65728736, 0.90539426, -0.06596833],
            [-0.043023292, 0.73763216, -0.33],
            [0.25863007, 1.0082227, -0.2867443],
            [-0.025555665, 0.7587344, -0.33],
            [-0.069979645, 0.7600795, -0.250762],
            [-0.07311567, 0.74933094, -0.27752763],
            [-0.19167922, 0.7947583, 0.7977194],
            [0.25555137, 1.007677, -0.2839787],
            [-0.18506202, 0.7937174, 0.67],
            [0.23232096, 0.97863513, -0.19446026],
            [-0.1497203, 0.75323594, 0.6849049],
            [-0.034343418, 0.77509636, -0.24178046],
            [0.6716609, 0.8471465, 0.023077901],
            [0.64070773, 0.8585589, -0.035566427],
            [-0.18895584, 0.7574538, 0.6752016],
            [0.28189385, 1.0011855, -0.28913078],
            [0.6298868, 0.59095716, 0.06],
            [0.24334021, 1.0004104, -0.19],
            [-0.049908306, 0.71719754, -0.19974335],
            [-0.13699242, 0.7798593, 0.73699963],
            [0.2519696, 1.0066084, -0.2118578],
            [0.6617275, 0.84652704, 0.046603516],
            [-0.1642625, 0.7656435, 0.67],
            [0.23522502, 0.96557266, -0.24269098],
            [0.65923744, 0.846894, 0.0119282575],
            [-0.19644615, 0.78313947, 0.7946282],
            [-0.19226807, 0.79386413, 0.689],
            [0.26807642, 0.9490079, -0.19437097],
            [0.6570685, 0.9053363, -0.07485582],
            [0.28873476, 0.964228, -0.28653178],
            [0.68564, 0.89799464, 0.042131092],
            [-0.05458957, 0.77422637, -0.29348606],
            [0.6423955, 0.8564724, 0.034728717],
            [-0.024825113, 0.7701223, -0.27782452],
            [0.2531273, 1.0070074, -0.25141865],
            [0.6915472, 0.8900535, 0.010912775],
            [-0.17966467, 0.75065446, 0.7471931],
            [-0.06290179, 0.7553271, -0.33],
            [0.23263918, 0.9740888, -0.21316141],
            [-0.15076376, 0.75253767, 0.73876435],
            [-0.1502795, 0.7806873, 0.67],
            [0.26234612, 1.0084505, -0.19612797],
            [-0.030860625, 0.7737706, -0.19153796],
            [0.6947699, 0.87859255, -0.042501792],
            [-0.16522959, 0.80776614, 0.79527926],
            [-0.0362733, 0.7756267, -0.3003113],
            [0.6642717, 0.9063582, 0.048397314],
            [0.27534837, 0.95142704, -0.27602008],
            [-0.18504252, 0.75390434, 0.7652121],
            [-0.15307072, 0.8044246, 0.69847685],
            [-0.031796273, 0.77417594, -0.19681251],
            [0.61663157, 0.6086687, -0.039047323],
            [-0.19614342, 0.771021, 0.8025552],
            [-0.17876464, 0.80537784, 0.67714363],
            [-0.15190129, 0.75184554, 0.7126036],
            [0.24950674, 0.95132476, -0.32129142],
            [-0.16962236, 0.77603745, 0.81],
            [-0.16046904, 0.8071116, 0.69109905],
            [-0.069031864, 0.7311037, -0.22963597],
            [-0.19055487, 0.7593338, 0.68677014],
            [-0.16605155, 0.764919, 0.81],
            [-0.13930756, 0.7660918, 0.7756839],
            [-0.15440914, 0.8050796, 0.71782136],
            [-0.07160646, 0.75625616, -0.30768767],
            [0.6471886, 0.90061206, -0.06407584],
            [0.25589076, 0.9491477, -0.32092893],
            [-0.042033464, 0.7764254, -0.19876283],
            [-0.018986786, 0.7288126, -0.20964104],
            [-0.13783552, 0.77046955, 0.75130105],
            [0.57896256, 0.56271106, -0.018678663],
            [-0.14914504, 0.8019792, 0.7322051],
            [0.23924556, 0.99478114, -0.19],
            [0.29190165, 0.9834455, -0.26668462],
            [-0.15460835, 0.7504578, 0.6934223],
            [-0.18039694, 0.7510102, 0.7483174],
            [-0.01330434, 0.74471813, -0.31266436],
            [-0.01937137, 0.7282952, -0.2820164],
            [0.6867137, 0.89690894, 0.039972484],
            [0.6941182, 0.88296205, 0.002017208],
            [-0.14417751, 0.7973757, 0.79900897],
            [-0.026294457, 0.77119625, -0.2261335],
            [-0.067212634, 0.72839487, -0.30208424],
            [-0.19691952, 0.7782452, 0.6854793],
            [0.27844223, 1.0037504, -0.2745498],
            [0.25179502, 0.95035756, -0.23151997],
            [-0.15722436, 0.8062031, 0.7225547],
            [-0.18515167, 0.7539875, 0.6982931],
            [-0.15951203, 0.74874365, 0.6825735],
            [-0.17089081, 0.7480776, 0.7949986],
            [0.6908996, 0.86632484, 0.06],
            [-0.19150314, 0.76061517, 0.69317347],
            [0.64806265, 0.8515024, 0.009797467],
            [0.63528794, 0.88145494, -0.027477283],
            [0.26389793, 0.94849205, -0.24567012],
            [0.29105383, 0.96982557, -0.28909603],
            [0.6631137, 0.90631336, -0.010791335],
            [0.23781057, 0.99574995, -0.22597605],
            [-0.17071056, 0.80757385, 0.79786724],
            [-0.1696276, 0.78338325, 0.67],
            [-0.1668503, 0.78622556, 0.81],
            [-0.014423382, 0.738157, -0.21689945],
            [0.6357211, 0.88352925, -0.040919494],
            [-0.046296027, 0.7166049, -0.22239669],
            [0.29232016, 0.97833306, -0.24735558],
            [0.5763852, 0.5659199, -0.062142298],
            [0.2910389, 0.969776, -0.29895842],
            [0.23535463, 0.9653034, -0.2711232],
            [-0.070059545, 0.75992167, -0.22351396],
            [0.6745556, 0.8479762, -0.038660098],
            [0.23730715, 0.9950137, -0.1918231],
            [-0.028580671, 0.7202838, -0.31966746],
            [-0.18673015, 0.8003453, 0.68183947],
            [-0.13800453, 0.76983, 0.75118655],
            [-0.19034198, 0.7965632, 0.6996239],
            [-0.1956127, 0.78658205, 0.77221763],
            [0.6123589, 0.6108278, 0.00817536],
            [-0.031068534, 0.7738639, -0.22469333],
            [-0.18380198, 0.8026149, 0.75930065],
            [0.68748355, 0.8566699, -0.076749705],
            [0.26148686, 0.9484621, -0.21033888],
            [0.6895479, 0.85932994, 0.02075481],
            [-0.18128785, 0.804151, 0.7264206],
            [-0.17738453, 0.8059307, 0.79857975],
            [-0.015480513, 0.7577912, -0.31315657],
            [-0.13887244, 0.788453, 0.7714736],
            [-0.19512923, 0.7880309, 0.8070556],
            [0.23263653, 0.9741069, -0.28167403],
            [-0.015500118, 0.75783914, -0.31889805],
            [-0.16179319, 0.7618575, 0.67],
            [0.2826539, 0.9563927, -0.2330482],
            [-0.070063315, 0.7329864, -0.32245535],
            [0.24905013, 1.005356, -0.26755738],
            [-0.13746099, 0.772156, 0.6969715],
            [-0.044719435, 0.77641445, -0.3122523],
            [0.6818045, 0.85456795, 0.06],
            [0.6055412, 0.55340266, 0.013684304],
            [-0.05763913, 0.75363874, -0.33],
            [0.2641747, 0.98107404, -0.33],
            [-0.023406833, 0.7239541, -0.29361725],
            [0.26337096, 1.0084321, -0.2324086],
            [0.244381, 0.9961269, -0.33],
            [0.66534615, 0.87659186, -0.08],
            [0.26145157, 0.971548, -0.19],
            [-0.07321901, 0.74790484, -0.22773996],
            [0.6014951, 0.58540165, 0.06],
            [0.23737708, 0.9837664, -0.19],
            [-0.17015961, 0.78562564, 0.81],
            [0.6349448, 0.87401605, -0.07202871],
            [-0.048638776, 0.7759631, -0.19868115],
            [-0.19451308, 0.76603407, 0.68784],
            [0.2599361, 1.0083556, -0.27313727],
            [0.2643091, 1.0083846, -0.23575774],
            [-0.03763686, 0.75423753, -0.33],
            [0.28989968, 0.9902566, -0.30584428],
            [-0.15330154, 0.80454344, 0.7463391],
            [0.24092413, 0.957422, -0.26727295],
            [-0.013755056, 0.7519088, -0.2260798],
            [0.2662178, 1.0081964, -0.24442396],
            [0.29080912, 0.9878523, -0.26996514],
            [0.2398831, 0.95853657, -0.2521617],
            [0.6919017, 0.88933915, -0.05643194],
            [-0.051130086, 0.775398, -0.2390455],
            [-0.1656942, 0.8077888, 0.7936506],
            [0.24049486, 0.9990332, -0.27071115],
            [-0.053233493, 0.71815866, -0.26926708],
            [0.6877483, 0.85697836, 0.058344983],
            [0.6480611, 0.9012242, -0.009260606],
            [0.23380427, 0.9877688, -0.28841934],
            [-0.13834004, 0.7869264, 0.6821488],
            [-0.15927649, 0.8068232, 0.7409918],
            [0.26083514, 0.97848046, -0.33],
            [0.2875686, 0.9946533, -0.2392021],
            [-0.1910429, 0.79565245, 0.7261889],
            [0.6871168, 0.89647144, -0.075393245],
            [-0.073046185, 0.74997777, -0.32428673],
            [0.28065673, 0.95470655, -0.21479885],
            [0.67859966, 0.89801335, -0.08],
            [-0.1459608, 0.78646743, 0.81],
            [0.27269042, 1.0066013, -0.24228033],
            [0.65415245, 0.90439063, 0.05984431],
            [-0.022849906, 0.7620642, -0.33],
            [-0.19605725, 0.7706604, 0.7287707],
            [0.27835968, 0.95309824, -0.2766384],
            [0.67934966, 0.85009897, -0.0004256786],
            [-0.019977035, 0.72752494, -0.27891827],
            [0.69289774, 0.86571115, -0.005366711],
            [0.69134843, 0.8622932, 0.015145173],
            [0.23528735, 0.99145883, -0.208926],
            [-0.15466893, 0.75043064, 0.6814916],
            [0.6597636, 0.8998004, -0.08],
            [0.28233668, 0.95610446, -0.30656868],
            [0.6393673, 0.860537, -0.058114525],
            [-0.16199745, 0.748221, 0.71802324],
            [-0.053020354, 0.77481616, -0.29576424],
            [0.6359484, 0.8843973, 0.039634008],
            [0.26098433, 0.9742272, -0.19],
            [-0.01922016, 0.76440465, -0.25269577],
            [-0.062044054, 0.76983714, -0.25979894],
            [0.67148626, 0.89710355, 0.06],
            [0.69450396, 0.8809256, -0.009652771],
            [0.6533571, 0.84865373, -0.031541914],
            [0.23638625, 0.9793911, -0.19],
            [0.2529734, 0.97809976, -0.33],
            [-0.026596235, 0.72150016, -0.24981064],
            [0.68860376, 0.85803646, -0.03375794],
            [0.6563353, 0.9051293, -0.035173833],
            [0.66266084, 0.8790174, -0.08],
            [0.26978076, 0.96904624, -0.19],
            [0.63751817, 0.8887257, -0.07222163],
            [0.5725116, 0.5736793, 0.008508466],
            [-0.068984576, 0.7459664, -0.19],
            [0.6947416, 0.87894464, 0.01894668],
            [0.6725942, 0.9053478, 0.036935717],
            [0.63487697, 0.8751604, -0.018115982],
            [-0.042734295, 0.7164548, -0.31292766],
            [-0.016611524, 0.76024026, -0.2467058],
            [0.63629365, 0.88554937, -0.06476676],
            [0.6362794, 0.8672227, -0.013144577],
            [0.25782883, 0.9487887, -0.19507933],
            [-0.03803029, 0.7759919, -0.23556212],
            [0.23233175, 0.9792755, -0.2936791],
            [-0.16745333, 0.74781865, 0.77431774],
            [0.59127426, 0.5546811, -0.021489382],
            [0.6574738, 0.8866802, -0.08],
            [0.661088, 0.9061267, -0.01216416],
            [-0.16850315, 0.7478556, 0.73055184],
            [0.2530583, 0.9499161, -0.2455326],
            [0.694288, 0.8821577, 0.043775752],
            [-0.042955387, 0.7764488, -0.3286212],
            [0.6452182, 0.899046, -0.04525437],
            [0.25954202, 0.9485795, -0.3126658],
            [0.6261676, 0.5667183, -0.02452719],
            [-0.19584352, 0.78578776, 0.77352756],
            [-0.015575184, 0.75802034, -0.2881238],
            [0.64933693, 0.8704006, -0.08],
            [-0.027929094, 0.7722406, -0.2969464],
            [-0.0490177, 0.7758915, -0.31784546],
            [-0.013503764, 0.7425895, -0.3295565],
            [0.25454196, 1.0074246, -0.2531511],
            [0.58854395, 0.55576986, -0.056872595],
            [-0.16708922, 0.7478144, 0.786626],
            [-0.16588978, 0.74783176, 0.80969495],
            [0.27211246, 0.9500936, -0.30480364],
            [-0.16018215, 0.748581, 0.7180017],
            [0.6620756, 0.906235, -0.013938692],
            [0.65039784, 0.8860348, -0.08],
            [-0.18964328, 0.79740405, 0.6940959],
            [0.6735062, 0.9050887, 0.020837775],
            [0.68641794, 0.8555085, 0.048172064],
            [-0.041997064, 0.77642393, -0.2749486],
            [0.63016325, 0.5759895, -0.044142377],
            [0.23670697, 0.9628316, -0.2020778],
            [0.6665905, 0.9063135, -0.00627371],
            [-0.15049729, 0.76494443, 0.67],
            [-0.044723514, 0.7764143, -0.32729036],
            [-0.029434208, 0.7302735, -0.19],
            [0.25581157, 0.9529819, -0.19],
            [0.65613943, 0.90507054, -0.010520581],
            [-0.018131122, 0.7300544, -0.24716575],
            [0.2749474, 0.9512374, -0.277651],
            [0.6649294, 0.8746445, -0.08],
            [-0.019779775, 0.7651304, -0.23072991],
            [0.6284409, 0.59520024, 0.050011374],
            [-0.060771693, 0.7220958, -0.31886977],
            [0.25058657, 1.0060606, -0.21286511],
            [-0.15433413, 0.77142096, 0.67],
            [0.64456904, 0.8542602, 0.0548522],
            [-0.021876259, 0.7254032, -0.2673411],
            [0.29094723, 0.96947813, -0.22804515],
            [-0.14216098, 0.7947509, 0.7698029],
            [0.6739638, 0.84778076, 0.033421613],
            [0.24203034, 1.0005484, -0.25525275],
            [0.57119274, 0.5863832, 0.004816559],
            [0.24422054, 1.0023754, -0.27688587],
            [-0.041458655, 0.7763965, -0.31245804],
            [-0.17017286, 0.74799055, 0.7920881],
            [-0.017553069, 0.7309762, -0.26051793],
            [0.6642349, 0.876489, 0.06],
            [0.2672092, 0.9488516, -0.21098824],
            [-0.14807929, 0.7544702, 0.75043994],
            [-0.016954988, 0.7608845, -0.27168113],
            [0.6361974, 0.88524455, -0.04267116],
            [-0.04015276, 0.7207244, -0.33],
            [-0.1965672, 0.7824182, 0.77880687],
            [-0.16152382, 0.7483037, 0.74587023],
            [0.6487859, 0.851029, -0.030013304],
            [-0.06976968, 0.7324169, -0.2988407],
            [-0.13715944, 0.77405185, 0.75413525],
            [-0.19276206, 0.7625718, 0.67409754],
            [0.23234062, 0.97955173, -0.32062006],
            [-0.053419054, 0.7746758, -0.3262913],
            [0.27298832, 0.9504114, -0.28285256],
            [0.2873481, 0.99499184, -0.30042467],
            [-0.045719106, 0.73019433, -0.19],
            [-0.07175388, 0.7370815, -0.25445372],
            [-0.16249044, 0.80748475, 0.6855025],
            [0.27692884, 1.0046535, -0.22678731],
            [0.25928476, 0.9635772, -0.19],
            [0.25576493, 1.0077256, -0.20537832],
            [0.2842824, 0.95801353, -0.28309307],
            [0.26884332, 1.0077329, -0.20866205],
            [0.63948125, 0.8923727, -0.07756029],
            [-0.15504622, 0.7502649, 0.79377925],
            [0.26797235, 0.98729676, -0.19],
            [-0.15292974, 0.80435073, 0.7243244],
            [-0.0731941, 0.7445509, -0.2025941],
            [-0.16459014, 0.7479048, 0.71145236],
            [0.266299, 0.97695607, -0.33],
            [0.6467119, 0.85247016, -0.032321777],
            [0.65786964, 0.90553975, -0.008756593],
            [0.6423171, 0.8843298, 0.06],
            [-0.17906946, 0.80524486, 0.80478764],
            [0.68446743, 0.8990633, 0.0123837],
            [0.60182786, 0.591072, 0.06],
            [-0.04306366, 0.71645087, -0.199433],
            [-0.054523095, 0.7186471, -0.2932648],
            [0.6654049, 0.8463689, -0.021672793],
            [-0.17816375, 0.8056283, 0.7857012],
            [-0.069006816, 0.76183885, -0.2756422],
            [-0.043871507, 0.71645665, -0.19977096],
            [-0.027679738, 0.7208098, -0.278447],
            [0.2627975, 1.0084468, -0.24530703],
            [-0.022633828, 0.7682401, -0.24169596],
            [-0.14500728, 0.7665324, 0.67],
            [0.64310586, 0.8644089, 0.06],
            [-0.014763803, 0.7558467, -0.32204813],
            [0.642325, 0.85655224, 0.044434614],
            [0.6442188, 0.89814085, -0.06407747],
            [0.68376464, 0.85307556, 0.011041149],
            [-0.14539163, 0.79870456, 0.7102638],
            [0.2902021, 0.98952353, -0.23347937],
            [0.2526511, 0.95521796, -0.33],
            [0.23236011, 0.9799935, -0.27841237],
            [0.664847, 0.8778583, -0.08],
            [0.68440557, 0.8991166, 0.045324586],
            [0.26295674, 0.9484573, -0.27323648],
            [-0.027853264, 0.77219534, -0.23016204],
            [-0.15889768, 0.80672073, 0.8099064],
            [0.6381422, 0.890022, 0.018799061],
            [-0.15766303, 0.8063492, 0.6868318],
            [0.23246507, 0.9755079, -0.30767256],
            [0.6746536, 0.90471774, -0.043172933],
            [0.6946885, 0.87949926, -0.076579206],
            [-0.072573155, 0.7400938, -0.21198763],
            [0.28777918, 0.99432033, -0.32975876],
            [-0.019218612, 0.728498, -0.2041884],
            [0.27071762, 0.94964975, -0.20315492],
            [-0.17100032, 0.7755964, 0.67],
            [0.5820283, 0.6062912, -0.075683124],
            [-0.03120635, 0.77392477, -0.19624853],
            [0.6776017, 0.90352017, -0.066002764],
            [0.28128493, 1.0016959, -0.25990075],
            [0.69063324, 0.8917056, 0.026094237],
            [-0.17086811, 0.74807453, 0.67010593],
            [0.6349807, 0.87359667, 0.02760025],
            [0.2573827, 1.0080414, -0.2470741],
            [-0.1444704, 0.7579168, 0.80968803],
            [0.65262246, 0.87748146, 0.06],
            [0.6350075, 0.87332124, 0.012888679],
            [0.6351969, 0.8808948, -0.068955876],
            [-0.07245324, 0.7533366, -0.29579005],
            [0.6397157, 0.85998935, -0.007017189],
            [0.6700937, 0.8468252, 0.030523533],
            [-0.02026895, 0.7271714, -0.23738398],
            [0.27635127, 1.0049672, -0.21221876],
            [-0.18973017, 0.79730284, 0.79342514],
            [-0.15766728, 0.8063506, 0.780663],
            [0.6377558, 0.8634893, -0.038116705],
            [0.29158378, 0.98505783, -0.3200732],
            [0.63651496, 0.8862111, 0.0160196],
            [-0.04660922, 0.71663845, -0.20838308],
            [0.62839967, 0.57082325, -0.014401949],
            [-0.029771473, 0.7732498, -0.23850964],
            [0.66538584, 0.84636855, -0.03211273],
            [0.23444468, 0.97241396, -0.19],
            [0.25251514, 0.9500982, -0.32783225],
            [-0.054735187, 0.7187341, -0.32230556],
            [0.6135004, 0.555783, -0.005877686],
            [0.5749601, 0.56817484, 0.037254788],
            [-0.0672976, 0.7643924, -0.20247978],
            [-0.026971627, 0.771647, -0.1917339],
            [-0.027393471, 0.77191466, -0.1920388],
            [-0.16988115, 0.80766773, 0.7039301],
            [-0.071895175, 0.75537777, -0.22253627],
            [0.68097824, 0.90166146, -0.07776185],
            [0.6638897, 0.8892831, -0.08],
            [0.28137085, 1.0016255, -0.22537309],
            [-0.036907602, 0.77577126, -0.26337656],
            [0.2740972, 1.0060424, -0.31071308],
            [-0.072518885, 0.75305206, -0.21027216],
            [0.6461436, 0.8529125, 0.033667743],
            [0.6074138, 0.61236626, -0.04486506],
            [0.6702156, 0.8570521, -0.08],
            [0.66096056, 0.9061103, 0.030182123],
            [0.28233343, 0.95610154, -0.32443872],
            [0.64838356, 0.8512886, -0.03412558],
            [-0.16192935, 0.7482324, 0.70449764],
            [0.5987735, 0.5531415, 0.05901051],
            [0.23987584, 0.95854473, -0.2569871],
            [-0.05040083, 0.72492945, -0.19],
            [-0.19628441, 0.78396904, 0.75682455],
            [0.29185894, 0.9732089, -0.21928473],
            [-0.17391546, 0.7838333, 0.67],
            [0.6771736, 0.8490106, -0.01284119],
            [0.29177144, 0.98416334, -0.25299096],
            [0.6236538, 0.56338173, 0.032322172],
            [-0.07310125, 0.7494766, -0.26076642],
            [-0.15041196, 0.752766, 0.7223025],
            [-0.071018144, 0.75781554, -0.307492],
            [0.66262996, 0.8464463, -0.00037484727],
            [-0.14150354, 0.7937473, 0.7395636],
            [0.26182136, 0.97702587, -0.33],
            [0.27535167, 0.9514286, -0.2658273],
            [0.68911874, 0.8940037, 0.027806757],
            [-0.14395244, 0.79711086, 0.7585932],
            [-0.06702446, 0.75539535, -0.19],
            [0.6474726, 0.85191125, 0.02012037],
            [0.23773837, 0.96125394, -0.30565205],
            [0.64534765, 0.85357016, -0.06632542],
            [0.25272235, 0.96152604, -0.19],
            [0.64224946, 0.85663843, -0.0645081],
            [-0.02182958, 0.76744986, -0.31833956],
            [-0.07316985, 0.744201, -0.28611147],
            [0.2343589, 0.98932046, -0.19499794],
            [0.6782073, 0.874048, -0.08],
            [-0.15429704, 0.80439687, 0.81],
            [0.26886472, 0.9491731, -0.3047546],
            [0.26910043, 1.0076743, -0.21070935],
            [-0.19538014, 0.7683182, 0.7228804],
            [-0.06602382, 0.7659836, -0.25550288],
            [0.67700183, 0.90379375, -0.057688486],
            [0.28352064, 0.95722437, -0.22641674],
            [-0.01326769, 0.7455538, -0.2240581],
            [0.63579124, 0.8838086, 0.0036477072],
            [-0.14266936, 0.7601567, 0.6808934],
            [-0.07256414, 0.75284815, -0.24115846],
            [-0.1838604, 0.80257505, 0.76573145],
            [0.2443744, 1.002491, -0.27234438],
            [-0.1666693, 0.80781287, 0.7625842],
            [0.23732673, 0.9618579, -0.2788537],
            [0.23486479, 0.9663597, -0.23011236],
            [-0.18805508, 0.7991076, 0.77318984],
            [-0.013280353, 0.74520016, -0.26181698],
            [0.6350812, 0.88005835, -0.039723672],
            [0.24870358, 1.0051823, -0.21206008],
            [0.6488858, 0.9017617, 0.010618723],
            [-0.18221474, 0.80362386, 0.70072794],
            [-0.18833017, 0.7567969, 0.6920576],
            [-0.029080985, 0.7728911, -0.23154059],
            [0.23827577, 0.9963909, -0.20672943],
            [-0.17689134, 0.7727776, 0.67],
            [0.2505179, 0.9808001, -0.19],
            [-0.015572969, 0.7580151, -0.23989117],
            [0.68829083, 0.85862446, 0.06],
            [0.6549811, 0.9046931, -0.06033998],
            [-0.03611424, 0.7274698, -0.33],
            [0.66578674, 0.8705582, 0.06],
            [-0.18339686, 0.80288583, 0.67136127],
            [-0.07295439, 0.75068164, -0.20209384],
            [0.23281631, 0.9838827, -0.23151572],
            [-0.02654236, 0.7215362, -0.29333442],
            [0.25994554, 1.0083565, -0.22449866],
            [0.66446286, 0.9063613, 0.01467262],
            [-0.014353022, 0.73840564, -0.19117336],
            [0.24661206, 0.98751825, -0.19],
            [0.6614126, 0.9061659, -0.05858773],
            [0.6674461, 0.84647614, 0.012335797],
            [0.69437903, 0.87105316, -0.068053104],
            [0.24115224, 0.9997087, -0.26207078],
            [0.659589, 0.84682924, -0.06532191],
            [-0.15535231, 0.75013494, 0.69242626],
            [0.66837233, 0.84657097, -0.041215617],
            [0.24299857, 0.9555013, -0.26825345],
            [-0.032016244, 0.7219385, -0.33],
            [0.23754479, 0.9938977, -0.19],
            [-0.016088126, 0.7337224, -0.3043573],
            [-0.13717608, 0.78170544, 0.7389879],
            [0.6227276, 0.60375255, -0.03182937],
            [-0.17290094, 0.74841565, 0.7475435],
            [-0.061567247, 0.77021235, -0.23755762],
            [0.66338503, 0.8463997, -0.07155182],
            [0.28257313, 0.95631856, -0.30895182],
            [0.6660815, 0.87717986, 0.06],
            [0.6795386, 0.85020417, 0.05431407],
            [0.6715976, 0.9055958, -0.02423486],
            [-0.02700321, 0.77166736, -0.29038483],
            [-0.16536018, 0.7800867, 0.81],
            [-0.07137498, 0.7359991, -0.21520293],
            [0.60364985, 0.5531747, 0.02930814],
            [0.25807124, 0.9792849, -0.33],
            [-0.16585673, 0.807795, 0.7179135],
            [-0.07105196, 0.75773257, -0.26392052],
            [0.27414307, 1.0060227, -0.24523418],
            [-0.13799587, 0.7698614, 0.7156582],
            [-0.072887674, 0.74177456, -0.3295974],
            [0.6460149, 0.89971197, 0.04437285],
            [0.27524635, 1.0055231, -0.28176713],
            [0.26919943, 1.0024468, -0.33],
            [-0.13942082, 0.7897994, 0.7449489],
            [0.6917559, 0.8896388, -0.026124882],
            [0.24668059, 0.95284986, -0.32928616],
            [0.2842868, 0.99888283, -0.31337333],
            [0.2830876, 0.9568006, -0.22499722],
            [0.29051656, 0.9815358, -0.33],
            [-0.14160253, 0.7617238, 0.72763985],
            [-0.054181363, 0.7407534, -0.33],
            [0.6542862, 0.8981134, 0.06],
            [0.26087022, 1.0084155, -0.3250315],
            [-0.048133336, 0.7168497, -0.2591557],
            [0.6860179, 0.8551026, -0.07387824],
            [-0.04932593, 0.77582943, -0.2853558],
            [-0.13694586, 0.7789948, 0.7137406],
            [-0.14911026, 0.7820075, 0.81],
            [0.23236334, 0.976846, -0.2506807],
            [-0.1477877, 0.8009193, 0.68559295],
            [-0.02925567, 0.7199166, -0.29731053],
            [-0.19551504, 0.7687325, 0.8009244],
            [-0.06072882, 0.72206503, -0.19195692],
            [-0.045752175, 0.7763461, -0.3204083],
            [0.27465773, 1.0057963, -0.22403498],
            [0.69408214, 0.88312, -0.022102123],
            [0.2792242, 0.95366627, -0.27400056],
            [0.63513076, 0.87228966, 0.0036621185],
            [0.2682583, 1.0078571, -0.2969569],
            [-0.02549169, 0.7706265, -0.22119264],
            [0.29183018, 0.9838518, -0.30587035],
            [0.6442495, 0.85455775, 0.032659005],
            [0.6745951, 0.9047379, 0.029895263],
            [0.6681411, 0.8889795, -0.08],
            [-0.023802841, 0.7236108, -0.19876638],
            [0.2834355, 0.9571397, -0.22672269],
            [0.68359405, 0.8997896, 0.0017213913],
            [0.5730649, 0.57214075, 0.0011063714],
            [-0.19419836, 0.7903053, 0.67743236],
            [0.6350241, 0.879565, 0.05878622],
            [-0.15567394, 0.7500027, 0.8063712],
            [-0.18884122, 0.7573304, 0.8027234],
            [0.69026065, 0.86041266, -0.073993206],
            [-0.06992791, 0.7601805, -0.2527983],
            [0.64357615, 0.8599524, -0.08],
            [-0.17708096, 0.74958616, 0.725142],
            [-0.17362465, 0.8070558, 0.7242187],
            [-0.020331634, 0.76580364, -0.22122972],
            [-0.0292906, 0.71989816, -0.21823086],
            [0.65837014, 0.9045293, -0.08],
            [-0.1808723, 0.8016071, 0.67],
            [-0.034956746, 0.7176206, -0.326072],
            [-0.15542705, 0.75010383, 0.75791186],
            [0.65941477, 0.90586686, -0.04002786],
            [0.6945811, 0.8803923, -0.015854739],
            [0.6525798, 0.8853858, -0.08],
            [-0.035874885, 0.71737206, -0.26801208],
            [0.6535938, 0.90417093, 0.031074394],
            [-0.036519464, 0.717216, -0.30706856],
            [-0.19038795, 0.7591223, 0.72299576],
            [-0.19323927, 0.7634114, 0.79389817],
            [0.26733586, 0.9988905, -0.19],
            [0.24587487, 1.0035414, -0.2176585],
            [0.6740708, 0.8905367, 0.06],
            [0.25336993, 1.0070843, -0.21387614],
            [0.69431037, 0.8706846, -0.07315807],
            [-0.14071572, 0.79241526, 0.73698896],
            [0.24482466, 0.9540805, -0.2712796],
            [0.65746444, 0.84728783, -0.071617156],
            [0.27918857, 1.0032592, -0.2541837],
            [-0.022772146, 0.7245304, -0.2280013],
            [0.27377662, 1.006177, -0.3174599],
            [-0.030270068, 0.7734949, -0.19092172],
            [-0.07265615, 0.75241107, -0.20356935],
            [0.57870656, 0.60312426, 0.017199466],
            [-0.16440906, 0.8077085, 0.7890071],
            [-0.18313207, 0.80305785, 0.7840549],
            [0.28840676, 0.96363497, -0.31425267],
            [-0.016187435, 0.759388, -0.20096466],
            [0.24119261, 0.9571523, -0.22456858],
            [-0.017469775, 0.7311154, -0.2349771],
            [-0.15227032, 0.8039924, 0.7230541],
            [0.23429246, 0.98914796, -0.23666775],
            [0.29221192, 0.9809996, -0.19899333],
            [-0.027438434, 0.72095793, -0.24868062],
            [0.6476292, 0.90092695, 0.005065912],
            [-0.18514039, 0.75397885, 0.80394304],
            [0.68429124, 0.8992144, 0.04066036],
            [-0.06456403, 0.7675665, -0.24153976],
            [-0.06095295, 0.77067333, -0.29898277],
            [-0.15563788, 0.7500173, 0.7714479],
            [-0.03588361, 0.7173698, -0.19394259],
            [0.26147264, 1.0084386, -0.2645338],
            [0.6414562, 0.8575862, 0.0146593675],
            [-0.04588772, 0.77633446, -0.19812466],
            [0.66870344, 0.846612, 0.047210813],
            [-0.19142984, 0.76051086, 0.7242916],
            [0.6486187, 0.89898807, 0.06],
            [-0.06921202, 0.73141044, -0.24884841],
            [0.25765264, 0.94881594, -0.24513692],
            [-0.027782362, 0.7430938, -0.19],
            [-0.03185972, 0.7742021, -0.321583],
            [-0.17981736, 0.7665566, 0.81],
            [0.6590356, 0.8968005, -0.08],
            [-0.14930111, 0.8020932, 0.76285195],
            [0.26800653, 0.94899434, -0.30399805],
            [-0.15017936, 0.802707, 0.74430466],
            [-0.07289626, 0.7418293, -0.2564319],
            [0.6460386, 0.89973104, 0.041257147],
            [-0.0382475, 0.716871, -0.26728684],
            [-0.15557215, 0.75004405, 0.8063559],
            [-0.14117847, 0.7932165, 0.7118059],
            [-0.05260417, 0.7584773, -0.19],
            [-0.19511592, 0.78806764, 0.7298135],
            [0.69481474, 0.8778742, 0.045097258],
            [0.635552, 0.8828029, 0.053550597],
            [-0.18847674, 0.7823739, 0.81],
            [0.65142924, 0.90319306, -0.02037762],
            [0.29025012, 0.9894019, -0.19500193],
            [-0.01327765, 0.7476339, -0.26651573],
            [0.28541884, 0.9593074, -0.32287422],
            [0.6348565, 0.87589115, -0.00740632],
            [0.65792656, 0.84717435, -0.04303422],
            [-0.1482305, 0.80127895, 0.77136046],
            [-0.021086987, 0.72623616, -0.19981772],
            [0.68734235, 0.8565088, 0.0049126595],
            [0.68984663, 0.85977143, -0.011836512],
            [0.27189016, 1.0068833, -0.2624995],
            [0.24783887, 0.9521773, -0.20265299],
            [0.6756944, 0.84839135, -0.04591265],
            [0.2685092, 0.94909585, -0.20376189],
            [0.63485587, 0.87679225, -0.03551334],
            [-0.13700023, 0.77997065, 0.72461706],
            [0.6930458, 0.8661094, 0.037645303],
            [-0.1686813, 0.74786556, 0.75233203],
            [0.24940822, 1.0055296, -0.19523534],
            [-0.17864929, 0.8054271, 0.7182728],
            [-0.15665925, 0.80600375, 0.8075058],
            [0.6756706, 0.8820071, -0.08],
            [-0.06589811, 0.7267714, -0.31283215],
            [-0.18897931, 0.7740561, 0.67],
            [-0.16232866, 0.8074601, 0.6915941],
            [0.6446613, 0.85417587, -0.0014698246],
            [-0.14624225, 0.79954696, 0.76144487],
            [-0.070778556, 0.7345165, -0.3296771],
            [-0.17122689, 0.74812436, 0.6736472],
            [-0.19185488, 0.7611292, 0.7443496],
            [0.23246224, 0.98136437, -0.19103992],
            [-0.15963987, 0.74871135, 0.67977405],
            [-0.06828145, 0.7629924, -0.24265629],
            [-0.018711746, 0.72921586, -0.33],
            [0.2626444, 0.97857636, -0.33],
            [-0.18937205, 0.7977143, 0.76762134],
            [-0.18558386, 0.8013035, 0.67878443],
            [-0.17235291, 0.7483095, 0.7426299],
            [0.2704291, 1.007334, -0.32981735],
            [-0.1944429, 0.765871, 0.76360697],
            [-0.042948764, 0.747082, -0.33],
            [-0.13719824, 0.7737567, 0.7926749],
            [0.2362843, 0.9933543, -0.29604882],
            [0.65472823, 0.88014656, -0.08],
            [-0.070194885, 0.73325163, -0.23174895],
            [0.6366466, 0.86614555, 0.039921384],
            [0.5746287, 0.56877047, -0.00987187],
            [0.6593897, 0.8468655, 0.035608202],
            [-0.15595214, 0.74989176, 0.6808631],
            [-0.06649583, 0.72748107, -0.3285541],
            [-0.15841736, 0.80658305, 0.68605494],
            [0.64507645, 0.8989225, -0.023556694],
            [-0.16091542, 0.7895593, 0.81],
            [-0.14929064, 0.7535423, 0.7433256],
            [-0.070195116, 0.7332521, -0.21941283],
            [-0.15534307, 0.7501388, 0.723511],
            [0.25456765, 0.98267895, -0.33],
            [-0.17577852, 0.7491509, 0.752993],
            [0.23823985, 0.99634266, -0.20038761],
            [-0.046347372, 0.7762904, -0.28881955],
            [-0.028697113, 0.72021884, -0.29695582],
            [0.6772436, 0.8727136, -0.08],
            [0.5834561, 0.55872846, -0.07730791],
            [0.24589363, 0.95334756, -0.1936032],
            [-0.02083408, 0.726517, -0.26213208],
            [-0.15219907, 0.8039524, 0.80133617],
            [0.6661797, 0.8767649, 0.06],
            [-0.16353537, 0.80762213, 0.77516526],
            [-0.17009193, 0.7710934, 0.67],
            [-0.1414558, 0.79367083, 0.731564],
            [-0.14013489, 0.79132015, 0.7135103],
            [0.25697425, 1.0079703, -0.286],
            [-0.18479195, 0.7777032, 0.67],
            [0.6767443, 0.8488213, -0.047638185],
            [0.66384864, 0.89214045, -0.08],
            [0.23561507, 0.99211895, -0.2552769],
            [-0.025475182, 0.72228616, -0.2491531],
            [0.68166304, 0.85151595, -0.014753217],
            [-0.13903306, 0.76676077, 0.7875932],
            [0.2922816, 0.97997606, -0.2300535],
            [-0.071508095, 0.73636454, -0.22195697],
            [0.23330434, 0.98607075, -0.27907187],
            [-0.18079475, 0.80441403, 0.71327674],
            [-0.020716565, 0.76625055, -0.30485025],
            [-0.027012244, 0.7716732, -0.30560556],
            [-0.01368812, 0.75153375, -0.24584967],
            [-0.035427928, 0.7502176, -0.19],
            [0.64872044, 0.90165704, -0.024297068],
            [0.69022363, 0.8603538, -0.0027099065],
            [0.6871175, 0.8964706, 0.05610961],
            [0.6948268, 0.87511474, 0.016937386],
            [-0.06445891, 0.76767206, -0.30796927],
            [0.63812596, 0.8627375, -0.03124129],
            [-0.02504011, 0.77028817, -0.28463328],
            [0.29209223, 0.9747577, -0.19374187],
            [0.24221802, 0.9976422, -0.33],
            [-0.019069826, 0.7286989, -0.24986108],
            [-0.030773433, 0.7737308, -0.26330373],
            [0.6539453, 0.848417, 0.014402684],
            [-0.158118, 0.80649287, 0.8023477],
            [-0.073169366, 0.7441945, -0.31495148],
            [0.6298063, 0.5914637, -0.029284624],
            [0.6162626, 0.5922573, -0.08],
            [0.6490917, 0.90189004, 0.049519967],
            [0.6718078, 0.9055465, -0.0049143797],
            [0.23987354, 0.95854735, -0.31818917],
            [0.23254585, 0.9747796, -0.28923717],
            [-0.13730365, 0.7825802, 0.77688223],
            [-0.05907552, 0.7317938, -0.19],
            [0.2450941, 0.9538893, -0.31696844],
            [0.66569525, 0.8682823, 0.06],
            [-0.17580348, 0.75714517, 0.81],
            [-0.028792702, 0.7627151, -0.33],
            [-0.05109539, 0.7174931, -0.26201257],
            [0.57967156, 0.5619688, 0.002429164],
            [0.6553235, 0.8479175, 0.006375119],
            [-0.14388417, 0.7585986, 0.69896394],
            [0.27259824, 0.96776456, -0.19],
            [-0.18459065, 0.75356853, 0.69919336],
            [-0.026007691, 0.72190326, -0.20010418],
            [-0.14089866, 0.762889, 0.6964419],
            [0.63822013, 0.86255443, -0.06352334],
            [0.28875467, 0.9926361, -0.32977998],
            [-0.01533878, 0.75743777, -0.31559992],
            [0.6944699, 0.87158626, -0.011225284],
            [0.67056346, 0.8469124, -0.001812801],
            [0.6854774, 0.8545779, 0.014078328],
            [0.281851, 0.95567876, -0.26615617],
            [0.29096928, 0.9873523, -0.32144102],
            [0.24201429, 1.0005336, -0.2297933],
            [-0.17691448, 0.7495268, 0.68595463],
            [-0.1545186, 0.8051295, 0.7557543],
            [-0.13755646, 0.7716797, 0.73614275],
            [0.6355512, 0.8699281, 0.044943176],
            [0.67350686, 0.84763914, 0.0048520607],
            [-0.1955665, 0.7867318, 0.6764344],
            [0.67972827, 0.90241605, -0.054260742],
            [0.28313592, 1.0000541, -0.3289438],
            [-0.18143575, 0.8040698, 0.7765112],
            [0.24368829, 1.0019633, -0.2762733],
            [0.6502626, 0.85015076, -0.020494414],
            [-0.03976203, 0.7762463, -0.21348174],
            [0.24095362, 0.9995091, -0.20099536],
            [-0.19690502, 0.7767865, 0.76038295],
            [-0.01346864, 0.75003004, -0.2549844],
            [-0.05143733, 0.7592967, -0.33],
            [0.67952925, 0.9025287, 0.059510365],
            [-0.13697338, 0.77606934, 0.70038366],
            [0.29152486, 0.98531353, -0.22767727],
            [-0.1523541, 0.7515888, 0.73585975],
            [0.65469927, 0.90459335, 0.006605441],
            [0.29132828, 0.97079927, -0.22296627],
            [-0.14317681, 0.796148, 0.75886697],
            [0.64936954, 0.8588321, 0.06],
            [0.63637626, 0.8858024, -0.071185544],
            [0.68323505, 0.8526553, 0.018215256],
            [0.24245426, 0.9559709, -0.26038295],
            [-0.019137293, 0.7642932, -0.30948153],
            [0.60727274, 0.5537196, -0.0573379],
            [0.25396666, 0.9496371, -0.2978815],
            [0.6789413, 0.8498777, -0.07925279],
            [-0.05735075, 0.7402041, -0.33],
            [0.6133678, 0.5557226, -0.02338472],
            [0.23909675, 0.9594595, -0.3044904],
            [0.65514535, 0.8479778, -0.01872001],
            [0.27137193, 1.0070525, -0.30149186],
            [0.6706404, 0.9058003, -0.04543181],
            [0.26088783, 0.97882074, -0.33],
            [-0.04327869, 0.7381021, -0.33],
            [0.6377352, 0.8891949, 0.0010123814],
            [0.66034925, 0.90602386, 0.029101638],
            [0.6725476, 0.84736747, -0.054429814],
            [-0.015798632, 0.7343595, -0.22254395],
            [0.25753173, 0.94883525, -0.2566788],
            [0.6887111, 0.8945512, -0.008918163],
            [-0.013689912, 0.7515441, -0.21403022],
            [0.26093373, 0.94848263, -0.29853073],
            [-0.13754547, 0.7796906, 0.81],
            [0.6664378, 0.84640574, 0.0024786699],
            [0.28974348, 0.96628606, -0.2279665],
            [-0.19177264, 0.794621, 0.8029319],
            [-0.013870638, 0.74040043, -0.22490902],
            [0.25223812, 1.0067056, -0.3177133],
            [-0.13883778, 0.7672668, 0.6837075],
            [0.63539267, 0.8706981, 0.03991609],
            [0.27215487, 1.0067928, -0.2754043],
            [-0.026173338, 0.72178775, -0.24197629],
            [0.63848764, 0.89067745, -0.018088946],
            [0.6349774, 0.8790952, -0.05924078],
            [0.23358184, 0.9870584, -0.22947524],
            [0.29178235, 0.97279423, -0.26759022],
            [0.616614, 0.6086794, -0.00037891397],
            [-0.062268823, 0.7232458, -0.29479143],
            [0.23456687, 0.989841, -0.28776637],
            [0.2482326, 1.004937, -0.19862002],
            [0.2778213, 1.0041356, -0.25132456],
            [-0.18694945, 0.7554773, 0.70558107],
            [0.64718837, 0.86155283, -0.08],
            [-0.1960242, 0.7705271, 0.737139],
            [-0.18909222, 0.75760233, 0.7715538],
            [-0.17953873, 0.8050322, 0.79013777],
            [0.28833145, 0.9635032, -0.20019946],
            [0.66730434, 0.9062635, -0.029748669],
            [-0.030860063, 0.74031574, -0.19],
            [-0.18129869, 0.8041451, 0.70178604],
            [-0.04067169, 0.7431131, -0.33],
            [-0.17779057, 0.7498517, 0.7270223],
            [-0.16931234, 0.7479093, 0.7546234],
            [-0.1522862, 0.80013365, 0.67],
            [0.57359296, 0.5952409, 0.026290294],
            [0.23294447, 0.98453784, -0.2652582],
            [0.2764699, 0.951997, -0.30056578],
            [-0.16515516, 0.77342486, 0.81],
            [-0.13843092, 0.78720665, 0.69502836],
            [0.289508, 0.9911327, -0.22742696],
            [-0.15397939, 0.7507497, 0.6960204],
            [-0.15017144, 0.80270165, 0.6881354],
            [-0.16704345, 0.7478142, 0.7606111],
            [-0.15768503, 0.8063563, 0.6833648],
            [-0.17996514, 0.8048305, 0.78478354],
            [0.26084334, 0.9484869, -0.24901983],
            [0.63521755, 0.8810279, -0.059176896],
            [-0.068233155, 0.7630652, -0.25679973],
            [0.23390935, 0.96881664, -0.24993883],
            [0.6367431, 0.8658831, 0.056682453],
            [0.67257684, 0.9053524, -0.011545551],
            [0.27628532, 0.96862507, -0.19],
            [-0.037040405, 0.71710086, -0.2768117],
            [-0.038172483, 0.7760167, -0.3049942],
            [-0.058157526, 0.7724867, -0.21744779],
            [0.2648548, 0.980459, -0.33],
            [-0.033071883, 0.71823114, -0.28747383],
            [-0.015463479, 0.75774944, -0.23892859],
            [-0.13885519, 0.78710896, 0.81],
            [0.57461625, 0.5973228, -0.018217811],
            [0.6656796, 0.88948774, -0.08],
            [0.6907942, 0.86129594, 0.054933723],
            [-0.037492517, 0.7758918, -0.27323914],
            [-0.15218425, 0.80394405, 0.7565211],
            [-0.06726315, 0.76443845, -0.32184994],
            [0.27247545, 0.9556635, -0.33],
            [0.6505999, 0.8499658, 0.020742537],
            [0.58008224, 0.5615612, 0.045282833],
            [-0.02800211, 0.7722838, -0.24750754],
            [-0.065757915, 0.7662893, -0.2940315],
            [0.68920046, 0.89389074, -0.027550355],
            [-0.15880424, 0.7489333, 0.76321656],
            [-0.049121134, 0.7170295, -0.31832165],
            [-0.05324215, 0.77473885, -0.2829748],
            [-0.028072728, 0.7723254, -0.30529875],
            [0.23235154, 0.9798172, -0.3285531],
            [0.6776878, 0.8492481, 0.025124416],
            [0.6380555, 0.8898511, -0.04417646],
            [0.6733415, 0.84758985, 0.057250973],
            [0.26684326, 0.9594894, -0.33],
            [-0.07303701, 0.7500544, -0.3036126],
            [-0.17151582, 0.78079545, 0.81],
            [0.6359958, 0.88456595, -0.049875587],
            [-0.14463355, 0.7577342, 0.7180142],
            [-0.14973788, 0.8024043, 0.7960138],
            [0.2428356, 0.95563954, -0.3291986],
            [-0.038989462, 0.7761456, -0.29829207],
            [-0.14481598, 0.7575335, 0.74923927],
            [-0.19653973, 0.7825917, 0.7099043],
            [-0.040147047, 0.7762889, -0.25159606],
            [-0.056148585, 0.7735379, -0.29929933],
            [-0.18946514, 0.7976088, 0.76164556],
            [0.2327605, 0.97333074, -0.20945694],
            [-0.070078515, 0.7598839, -0.27301997],
            [-0.1943581, 0.76567745, 0.6828792],
            [0.27509645, 0.951307, -0.20577112],
            [-0.19648409, 0.7727033, 0.7875138],
            [-0.1965648, 0.7824337, 0.6826186],
            [0.289109, 0.99195504, -0.25715557],
            [-0.17934334, 0.80512196, 0.78304696],
            [-0.025907435, 0.77092654, -0.19906166],
            [-0.19388752, 0.7909629, 0.70281607],
            [-0.15197502, 0.7946351, 0.81],
            [0.26305494, 1.0084416, -0.32897013],
            [-0.06632505, 0.76562685, -0.29056725],
            [-0.05087163, 0.7754671, -0.1982079],
            [-0.047025483, 0.7166883, -0.26956168],
            [0.69318545, 0.8862259, -0.000599444],
            [0.6443062, 0.8706987, 0.06],
            [-0.15508305, 0.80537885, 0.6762629],
            [0.27029294, 0.9495672, -0.33],
            [0.6828224, 0.85234106, -0.070644006],
            [-0.13710164, 0.7810865, 0.6803923],
            [0.25529686, 1.0076168, -0.32527822],
            [0.5864536, 0.6092912, -0.03338011],
            [0.61764884, 0.6080197, 0.035204835],
            [-0.19626291, 0.7840707, 0.7913763],
            [-0.1680325, 0.7748394, 0.81],
            [-0.024282923, 0.752128, -0.19],
            [0.634875, 0.87751776, 0.033192754],
            [0.6477394, 0.8517238, 0.001199825],
            [0.24984296, 0.9685245, -0.33],
            [0.24695669, 0.9526832, -0.21719116],
            [-0.035336383, 0.71751404, -0.24431439],
            [0.64341104, 0.85538167, 0.032755494],
            [-0.013927519, 0.7401304, -0.27688274],
            [-0.06580214, 0.76623905, -0.19950166],
            [0.61037457, 0.55455786, -0.04291427],
            [-0.14682986, 0.7555366, 0.77504355],
            [-0.14143515, 0.7619903, 0.7524074],
            [0.6724482, 0.84734124, 0.014889728],
            [-0.15131271, 0.78854007, 0.67],
            [0.23538415, 0.9722922, -0.19],
            [-0.014233028, 0.7388499, -0.24055023],
            [-0.18191673, 0.7518298, 0.7033451],
            [0.23910806, 0.9910364, -0.19],
            [-0.15984307, 0.80696666, 0.6978472],
            [-0.03380406, 0.7179776, -0.19573581],
            [-0.16491225, 0.74825495, 0.81],
            [0.64209986, 0.8568112, 0.019048668],
            [0.66772586, 0.88190717, -0.08],
            [0.27056837, 0.9846958, -0.33],
            [0.6861289, 0.8975139, -0.03658262],
            [0.2383489, 0.9604126, -0.23528093],
            [0.23361275, 0.9871609, -0.21750621],
            [-0.19372617, 0.79128873, 0.7611387],
            [0.28793803, 0.99406266, -0.25922793],
            [-0.024597598, 0.7699434, -0.21918866],
            [0.23353928, 0.9699861, -0.28683913],
            [-0.04042391, 0.7165841, -0.29560372],
            [-0.13704139, 0.7751471, 0.71232843],
            [-0.06399392, 0.7681267, -0.19206016],
            [0.635156, 0.8806182, -0.017224135],
            [0.6407624, 0.89424264, -0.068704896],
            [0.66371626, 0.8861145, 0.06],
            [-0.13816951, 0.7863731, 0.7424288],
            [-0.1839497, 0.77390844, 0.67],
            [-0.19519693, 0.7878421, 0.7673882],
            [0.2340583, 0.9683881, -0.26175332],
            [0.23314136, 0.98542076, -0.23134914],
            [-0.05065526, 0.7475183, -0.19],
            [-0.07250472, 0.7531145, -0.23188467],
            [0.6773971, 0.89564776, -0.08],
            [-0.17018987, 0.8076355, 0.7410018],
            [0.66160935, 0.8465397, -0.046024334],
            [-0.15644106, 0.80592334, 0.69344413],
            [0.27719584, 0.9523983, -0.23882876],
            [0.28029963, 0.954435, -0.20477529],
            [0.63883466, 0.8614288, -0.05582861],
            [-0.19621898, 0.7713546, 0.75292695],
            [-0.047009274, 0.7431425, -0.33],
            [0.28962046, 0.9660125, -0.27392808],
            [-0.19075936, 0.7595983, 0.7800742],
            [-0.03238731, 0.7184877, -0.22557233],
            [0.6379827, 0.86302227, -0.035874136],
            [0.639708, 0.8600011, -0.050531905],
            [-0.016793668, 0.732314, -0.20397577],
            [-0.18973061, 0.7973023, 0.80606675],
            [0.6746351, 0.9047241, 0.0345165],
            [-0.0728199, 0.7413635, -0.30439663],
            [-0.07259153, 0.75272137, -0.23984282],
            [0.6438851, 0.8978197, -0.058790136],
            [-0.06578172, 0.7662623, -0.23032118],
            [0.6357829, 0.88377595, -0.0034587714],
            [-0.15483811, 0.8052724, 0.6782313],
            [0.6679223, 0.84652126, 0.05033435],
            [-0.18717885, 0.7999428, 0.72955966],
            [0.26872793, 0.9491428, -0.31328574],
            [0.26829222, 1.0078502, -0.19298983],
            [-0.1955193, 0.786882, 0.6811967],
            [0.23711072, 0.9947131, -0.24546689],
            [-0.014948435, 0.7365115, -0.19185282],
            [0.6357362, 0.8691371, 0.057558406],
            [0.6502585, 0.850153, 0.011419982],
            [-0.18807907, 0.7565442, 0.8052483],
            [0.61963916, 0.5595448, 0.034406766],
            [-0.047511317, 0.74315983, -0.19],
            [-0.15874176, 0.7861879, 0.81],
            [-0.029583849, 0.77315456, -0.2135298],
            [-0.0141551765, 0.7391536, -0.24040088],
            [-0.13712867, 0.77430385, 0.7607303],
            [-0.15294994, 0.80436134, 0.7940225],
            [-0.06273411, 0.76926553, -0.2289746],
            [-0.15882969, 0.74892616, 0.7860545],
            [0.2765251, 0.9609137, -0.33],
            [0.271543, 0.9827742, -0.33],
            [0.692966, 0.8658925, 0.05290404],
            [-0.019215314, 0.7285024, -0.20724265],
            [-0.17878328, 0.8053698, 0.7914612],
            [-0.013959217, 0.7399851, -0.29349238],
            [0.29071268, 0.9687615, -0.30909672],
            [-0.04694939, 0.71667874, -0.31003755],
            [-0.017895956, 0.7304205, -0.21359867],
            [0.2574017, 0.972505, -0.33],
            [0.23238645, 0.98044014, -0.23232664],
            [0.2900095, 0.98999673, -0.23884982],
            [-0.1371623, 0.7740293, 0.8048982],
            [0.29220414, 0.9810891, -0.27568078],
            [-0.16219944, 0.8074398, 0.70968044],
            [0.2413111, 0.9570354, -0.28740498],
            [0.29218873, 0.9812581, -0.19962013],
            [-0.034248795, 0.7622664, -0.19],
            [0.2917195, 0.98442495, -0.2790788],
            [-0.15788902, 0.74920636, 0.7287504],
            [0.6731466, 0.9051946, 0.021446258],
            [0.68861955, 0.8580569, 0.053702645],
            [0.26226622, 0.9784168, -0.33],
            [-0.18445034, 0.7534669, 0.7907474],
            [0.6837638, 0.8996528, -0.026446266],
            [-0.15773639, 0.749255, 0.7285219],
            [-0.04265818, 0.74596715, -0.19],
            [0.648868, 0.8988133, 0.06],
            [0.26267058, 0.9484526, -0.2751745],
            [0.66170347, 0.8821808, -0.08],
            [-0.01616761, 0.7593465, -0.2686885],
            [-0.14977637, 0.7531968, 0.73664427],
            [-0.1586707, 0.8066567, 0.697233],
            [-0.14414711, 0.7973403, 0.77556247],
            [0.24770328, 0.9522525, -0.22309378],
            [-0.04326185, 0.7164503, -0.30331838],
            [0.25725758, 1.0080203, -0.29098347],
            [0.27195546, 1.0060993, -0.19],
            [0.57928354, 0.56236875, -0.008844601],
            [0.24862114, 0.9621682, -0.19],
            [-0.15419279, 0.8049792, 0.7832283],
            [0.6759153, 0.90424967, 0.0029934617],
            [-0.17402638, 0.80696076, 0.72817534],
            [-0.06278735, 0.76922, -0.1932362],
            [-0.023157736, 0.72417635, -0.31326523],
            [-0.17472601, 0.8067813, 0.73842657],
            [-0.14601512, 0.7993285, 0.78285587],
            [0.6740547, 0.8478099, -0.01022388],
            [0.6948521, 0.8761574, 0.041198686],
            [-0.056656804, 0.77329004, -0.32510749],
            [-0.06233104, 0.7696036, -0.25978845],
            [-0.041416373, 0.77639395, -0.19287015],
            [-0.15960947, 0.748719, 0.7805543],
            [0.6303728, 0.5891984, 0.058183532],
            [0.29231256, 0.9777642, -0.251025],
            [0.2495253, 0.9961416, -0.33],
            [0.59980804, 0.6130341, -0.014809828],
            [-0.06840721, 0.73010004, -0.3074144],
            [0.29168174, 0.97229326, -0.23701723],
            [0.66808623, 0.9061891, 0.0035086714],
            [0.68757534, 0.8567759, -0.056121156],
            [0.27897578, 1.0034025, -0.25437778],
            [-0.047547832, 0.77614146, -0.1990051],
            [0.24610361, 0.98687893, -0.19],
            [0.6380822, 0.86282367, -0.029596418],
            [0.29215792, 0.9753325, -0.22193033],
            [0.5819824, 0.5598627, 0.03472984],
            [0.6937372, 0.8682588, -0.028485028],
            [-0.052890938, 0.7748604, -0.2315093],
            [0.28889287, 0.96452564, -0.24809982],
            [-0.15283619, 0.8043012, 0.80861163],
            [-0.15795636, 0.7491852, 0.79460955],
            [-0.1827972, 0.7851889, 0.81],
            [0.26280272, 0.9995233, -0.33],
            [-0.14488563, 0.7981701, 0.783146],
            [0.6824506, 0.9006603, -0.06618678],
            [-0.014170279, 0.73909366, -0.20370054],
            [-0.07291182, 0.74193025, -0.28949866],
            [0.2323708, 0.98018855, -0.25546807],
            [0.6636845, 0.8463866, 0.011911578],
            [0.63494796, 0.8739761, 0.02660261],
            [0.693335, 0.8857854, 0.026872186],
            [0.26543063, 0.9486122, -0.20297067],
            [0.6864019, 0.85549194, 0.023438415],
            [0.6516527, 0.84942394, 0.05277062],
            [0.24364771, 1.001931, -0.2209125],
            [0.6473222, 0.9007088, -0.0025821126],
            [0.26109856, 0.96605843, -0.33],
            [-0.14575005, 0.7964346, 0.67],
            [0.29127538, 0.97060144, -0.2964218],
            [0.6841026, 0.89937353, -0.06888657],
            [0.6462493, 0.8528285, -0.043231066],
            [-0.038575593, 0.75847197, -0.33],
            [-0.14256577, 0.7953281, 0.78403616],
            [0.6350283, 0.87312347, 0.059303563],
            [-0.015096025, 0.7361008, -0.2594892],
            [-0.1901561, 0.79679304, 0.7736584],
            [0.28689605, 0.9956563, -0.24047068],
            [-0.19348508, 0.76386994, 0.67757726],
            [-0.06662666, 0.7652581, -0.3261437],
            [0.23757015, 0.9954042, -0.30019638],
            [-0.1960106, 0.7851549, 0.7382313],
            [-0.015456146, 0.7351692, -0.24227679],
            [0.6558482, 0.8477471, -0.0739002],
            [-0.038964152, 0.7167586, -0.21066192],
            [0.2687641, 1.002943, -0.19],
            [-0.14201581, 0.7945367, 0.770886],
            [0.26413026, 0.9485052, -0.27774087],
            [-0.020752998, 0.72660863, -0.26749843],
            [0.28693542, 0.9956, -0.22155046],
            [-0.16914825, 0.7478966, 0.7465841],
            [-0.14494966, 0.77249825, 0.81],
            [-0.071612395, 0.75623894, -0.28956082],
            [0.6381004, 0.88993984, -0.06105111],
            [0.29116815, 0.98668504, -0.23135968],
            [-0.16386306, 0.7479704, 0.6899092],
            [0.66181374, 0.90620947, -0.055975884],
            [0.57575214, 0.56686723, -0.04210505],
            [-0.031544413, 0.7416548, -0.19],
            [-0.040476695, 0.71657914, -0.24732599],
            [-0.01329271, 0.747968, -0.30127382],
            [0.66121787, 0.8714581, -0.08],
            [-0.14870541, 0.75397843, 0.69207615],
            [0.24114637, 0.95719826, -0.26986533],
            [0.2573244, 0.94886947, -0.30195948],
            [-0.14729396, 0.7551266, 0.6749986],
            [-0.19349705, 0.76389277, 0.69532704],
            [-0.15523352, 0.7501849, 0.7581211],
            [0.69173086, 0.8896894, -0.053734884],
            [-0.01347336, 0.75006914, -0.19084233],
            [0.6358437, 0.86871725, -0.009817577],
            [0.65618426, 0.90508413, -0.02564707],
            [-0.073117115, 0.7493159, -0.25190008],
            [0.29015523, 0.98964083, -0.3000842],
            [-0.19079603, 0.7959815, 0.72245044],
            [0.6107104, 0.61144584, -0.04139206],
            [-0.04685569, 0.71666723, -0.24874362],
            [-0.13715476, 0.78153884, 0.77722675],
            [0.2803214, 1.0024499, -0.27799383],
            [-0.1488742, 0.7822786, 0.81],
            [0.6865755, 0.89705503, -0.035080127],
            [-0.15811038, 0.8064905, 0.67996114],
            [0.58301526, 0.60706365, 0.04396292],
            [-0.06413783, 0.72491246, -0.22429079],
            [-0.19635108, 0.77198595, 0.792985],
            [-0.16728902, 0.7772984, 0.67],
            [0.2599266, 0.94854623, -0.21112767],
            [0.634864, 0.87718225, 0.012219586],
            [0.27294263, 0.9896962, -0.19],
            [0.65733856, 0.8803035, 0.06],
            [0.24921452, 1.0054364, -0.23931992],
            [0.67500746, 0.90459293, 0.02548858],
            [-0.07203161, 0.7549277, -0.3118813],
            [0.64609337, 0.8642822, -0.08],
            [0.2385992, 0.96008474, -0.24550903],
            [0.6632473, 0.8464068, -0.076083586],
            [0.27119523, 0.9507071, -0.19],
            [-0.01417928, 0.73905814, -0.31671378],
            [0.24026668, 0.9581126, -0.2508821],
            [-0.06362207, 0.7684765, -0.28027764],
            [-0.13703737, 0.78043544, 0.76565826],
            [-0.14097673, 0.78445876, 0.67],
            [-0.1642591, 0.7948923, 0.81],
            [-0.14961222, 0.75331193, 0.67870885],
            [-0.19207194, 0.7941697, 0.68722254],
            [0.63082266, 0.57972926, 0.027963292],
            [-0.13923988, 0.78937536, 0.7393693],
            [0.24409312, 1.0022784, -0.20448902],
            [0.66517144, 0.8463655, -0.058535784],
            [0.692611, 0.8649847, -0.055334337],
            [0.2447043, 0.95416737, -0.31648168],
            [-0.03099452, 0.7567386, -0.33],
            [0.6931322, 0.8663501, -0.00097279774],
            [0.6810864, 0.8511355, -0.049724597],
            [0.60091937, 0.61305803, -0.07100907],
            [-0.1933955, 0.7637006, 0.7528994],
            [0.676563, 0.87992394, 0.06],
            [-0.19397044, 0.7648365, 0.7007593],
            [-0.025920104, 0.76452667, -0.33],
            [0.6934593, 0.8854009, 0.0012496097],
            [-0.17220064, 0.80734605, 0.67262864],
            [0.2507142, 0.9850859, -0.19],
            [-0.16286778, 0.7860596, 0.81],
            [0.23910438, 0.997451, -0.20444913],
            [0.26627472, 1.0081888, -0.21555449],
            [-0.028390525, 0.7725092, -0.22736035],
            [-0.19682896, 0.7801826, 0.7556401],
            [-0.13969193, 0.79040325, 0.7162344],
            [-0.02004373, 0.7654574, -0.32946098],
            [-0.07318086, 0.7443526, -0.19933859],
            [0.6601312, 0.84673774, -0.079477146],
            [0.6463461, 0.8527523, -0.007150503],
            [-0.17821868, 0.805606, 0.7950816],
            [-0.16665362, 0.7478152, 0.7611576],
            [0.6888858, 0.8943197, 0.049487296],
            [-0.13751397, 0.77188677, 0.73763895],
            [-0.1765057, 0.78534216, 0.81],
            [-0.15697776, 0.80611765, 0.7395564],
            [-0.048287224, 0.7199813, -0.19],
            [0.23756891, 0.9954024, -0.3062349],
            [0.6900116, 0.86002266, -0.07186532],
            [-0.05147712, 0.7175992, -0.30583233],
            [0.61984813, 0.55971193, -0.038013183],
            [0.25124067, 0.95057154, -0.31279603],
            [-0.16474196, 0.75584215, 0.81],
            [-0.05451735, 0.7465622, -0.33],
            [0.67245436, 0.9053848, -0.07231432],
            [0.6365965, 0.882813, -0.08],
            [-0.0133894645, 0.74929494, -0.29020965],
            [-0.14388475, 0.7585979, 0.7553153],
            [-0.19488323, 0.78868616, 0.7349097],
            [0.24054803, 0.9990895, -0.2875931],
            [-0.17805614, 0.7742248, 0.67],
            [-0.061644174, 0.7227477, -0.20059797],
            [0.6924109, 0.8882193, -0.01881294],
            [-0.16261874, 0.778743, 0.81],
            [-0.036371488, 0.7172505, -0.27339014],
            [0.24830596, 0.9519252, -0.2332152],
            [0.63940966, 0.8922587, 0.002052464],
            [0.25314254, 1.0070122, -0.21338655],
            [0.26337346, 0.99123836, -0.19],
            [-0.13692261, 0.77780133, 0.6745793],
            [0.2601797, 0.94852704, -0.30404177],
            [0.2846799, 0.9984519, -0.3138248],
            [0.235466, 0.99182373, -0.30736375],
            [-0.021087248, 0.7666647, -0.32930058],
            [0.2654294, 0.97717035, -0.19],
            [-0.04898489, 0.7389052, -0.19],
            [-0.19090259, 0.7958406, 0.7434317],
            [0.23248748, 0.9816123, -0.20631708],
            [0.27843612, 0.95314676, -0.2716823],
            [-0.13846713, 0.78731585, 0.68804586],
            [-0.1744755, 0.7487803, 0.7203628],
            [0.281834, 1.0012369, -0.3251236],
            [0.2588371, 1.0082476, -0.2723442],
            [0.274141, 0.9508775, -0.21664883],
            [-0.015403429, 0.73529994, -0.22573046],
            [0.5729719, 0.572382, 0.014002107],
            [-0.024060512, 0.74794406, -0.33],
            [0.6612937, 0.90615195, -0.06920101],
            [0.26857713, 0.94911027, -0.2105706],
            [-0.13701126, 0.77550936, 0.7622826],
            [-0.06522763, 0.72602546, -0.26878148],
            [-0.18368968, 0.75293696, 0.74566114],
            [-0.06932492, 0.731607, -0.19868127],
            [-0.050834738, 0.7174238, -0.19156162],
            [-0.040005643, 0.77627385, -0.20138422],
            [-0.0732331, 0.7475777, -0.3134819],
            [0.6391055, 0.8618929, 0.06],
            [-0.14183906, 0.79427046, 0.75689465],
            [-0.028680263, 0.7577817, -0.33],
            [-0.024762584, 0.7228271, -0.22614017],
            [0.64803606, 0.90120727, -0.04757857],
            [0.6373467, 0.8643883, 0.026641436],
            [0.6548544, 0.90464866, 0.033224452],
            [0.23620106, 0.9636932, -0.27751234],
            [0.2579568, 0.9964993, -0.19],
            [-0.14693835, 0.7554392, 0.7591245],
            [-0.07102749, 0.75779265, -0.21909268],
            [-0.15624253, 0.8058485, 0.68909365],
            [0.6400668, 0.8594625, 0.037294284],
            [-0.041063655, 0.7165304, -0.1963665],
            [-0.030795896, 0.7437483, -0.19],
            [-0.071414724, 0.7567939, -0.23221208],
            [0.6628172, 0.8930654, -0.08],
            [0.665136, 0.87077034, 0.06],
            [0.28310004, 0.9568125, -0.25524864],
            [0.6938687, 0.8687429, -0.049335964],
            [-0.06989472, 0.7326558, -0.28943083],
            [0.23930243, 0.9592107, -0.2854756],
            [0.6813071, 0.85127884, -0.014217905],
            [0.6351394, 0.8722268, -0.016421169],
            [-0.16184159, 0.7482474, 0.8051032],
            [0.64504176, 0.8988921, -0.026227036],
            [0.6352264, 0.88108367, 0.017741438],
            [-0.042394668, 0.75683665, -0.33],
            [0.2343065, 0.9891847, -0.20531455],
            [0.24144448, 0.99999577, -0.2920543],
            [0.64070016, 0.85856915, -0.022462483],
            [-0.053976316, 0.7184318, -0.23811474],
            [0.6468448, 0.9003578, -0.06036144],
        ];

        let kdt: ImmutableDynamicKdTree<f64, usize, 3, 32> =
            ImmutableDynamicKdTree::new_from_slice(&points);

        assert_eq!(kdt.size(), points.len());
    }
}
