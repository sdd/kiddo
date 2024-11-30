//! Immutable Floating point k-d tree. Offers less memory utilisation, smaller size
//! when serialized, and faster more consistent query performance. This comes at the
//! expense of not being able to modify the contents of the tree after its initial
//! construction, and longer construction times - perhaps prohibitively so.
//! As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
//! values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled

pub use crate::float::kdtree::Axis;
use crate::float_leaf_slice::leaf_slice::{LeafSlice, LeafSliceFloat};
#[cfg(feature = "modified_van_emde_boas")]
use crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;
use crate::types::Content;
#[cfg(feature = "rkyv")]
use aligned_vec::CACHELINE_ALIGN;
use aligned_vec::{avec, AVec, ConstAlign};
use array_init::array_init;
use az::{Az, Cast};
use ordered_float::OrderedFloat;
#[cfg(feature = "rkyv")]
use rkyv::vec::ArchivedVec;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use std::fmt::Debug;

/// Immutable floating point k-d tree
///
/// Offers less memory utilisation, smaller size vs non-immutable tree
/// when serialized, and faster more consistent query performance. This comes at the
/// expense of not being able to modify the contents of the tree after its initial
/// construction, and longer construction times.
///
/// Compared to non-dynamic ImmutableKdTree, this can handle data like point clouds
/// that may have many occurrences of multiple points have the exact same value on a given axis.
/// This comes at the expense of slower performance. Memory usage should still be very efficient,
/// more so than the standard and non-dynamic immutable tree types.
///
/// As with the vanilla tree, [`f64`] or [`f32`] are supported currently for co-ordinate
/// values, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled
///
/// A convenient type alias exists for ImmutableKdTree with some sensible defaults set: [`kiddo::ImmutableKdTree`](`crate::ImmutableKdTree`).
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ImmutableKdTree<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    pub(crate) stems: AVec<A>,

    #[cfg_attr(feature = "serde", serde(with = "crate::custom_serde::array_of_vecs"))]
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "A: Serialize, T: Serialize",
            deserialize = "A: Deserialize<'de>, T: Deserialize<'de> + Copy + Default"
        ))
    )]
    pub(crate) leaf_points: [Vec<A>; K],
    pub(crate) leaf_items: Vec<T>,
    pub(crate) leaf_extents: Vec<(u32, u32)>,
    pub(crate) max_stem_level: isize,
}

#[cfg(feature = "rkyv")]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ImmutableKdTreeRK<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize> {
    pub(crate) stems: Vec<A>,
    pub(crate) leaf_points: [Vec<A>; K],
    pub(crate) leaf_items: Vec<T>,
    pub(crate) leaf_extents: Vec<(u32, u32)>,
    pub(crate) max_stem_level: isize,
}

#[cfg(feature = "rkyv")]
impl<A: Axis, T: Content, const K: usize, const B: usize> From<ImmutableKdTree<A, T, K, B>>
    for ImmutableKdTreeRK<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdTreeRK`, from an `ImmutableKdTree`
    ///
    ///
    /// `ImmutableKdTreeRK` implements `rkyv::Archive`, permitting it to be serialized to
    /// as close to a zero-copy form as possible. Zero-copy-deserialized `ImmutableKdTreeRK`
    /// instances can be converted to instances of `AlignedArchivedImmutableKdTree`, which involves
    /// a copy of the stems to ensure correct alignment, but re-use of the rest of the structure.
    /// `AlignedArchivedImmutableKdTree` instances can then be queried in the same way as the original
    /// `ImmutableKdTree`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = (&*points).into();
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    fn from(orig: ImmutableKdTree<A, T, K, B>) -> Self {
        let ImmutableKdTree {
            stems,
            leaf_points,
            leaf_items,
            leaf_extents,
            max_stem_level,
        } = orig;

        let (ptr, _, length, capacity) = stems.into_raw_parts();
        let stems = unsafe { Vec::from_raw_parts(ptr, length, capacity) };

        ImmutableKdTreeRK {
            stems,
            leaf_points,
            leaf_items,
            leaf_extents,
            max_stem_level,
        }
    }
}

#[cfg(feature = "rkyv")]
#[derive(Debug, PartialEq)]
pub struct AlignedArchivedImmutableKdTree<
    'a,
    A: Copy + Default,
    T: Copy + Default,
    const K: usize,
    const B: usize,
> {
    pub(crate) stems: AVec<A, ConstAlign<CACHELINE_ALIGN>>,
    pub(crate) leaf_points: &'a [ArchivedVec<A>; K],
    pub(crate) leaf_items: &'a ArchivedVec<T>,
    pub(crate) leaf_extents: &'a ArchivedVec<(u32, u32)>,
    pub max_stem_level: isize,
}

#[cfg(feature = "rkyv")]
impl<
        'a,
        A: Copy + Default + rkyv::Archive<Archived = A>,
        T: Copy + Default + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > AlignedArchivedImmutableKdTree<'a, A, T, K, B>
{
    pub fn new_from(
        value: &'a ArchivedImmutableKdTreeRK<A, T, K, B>,
    ) -> AlignedArchivedImmutableKdTree<'a, A, T, K, B> {
        AlignedArchivedImmutableKdTree {
            stems: AVec::from_slice(CACHELINE_ALIGN, &value.stems[..]),
            leaf_points: &value.leaf_points,
            leaf_extents: &value.leaf_extents,
            leaf_items: &value.leaf_items,
            max_stem_level: value.max_stem_level as isize,
        }
    }

    #[cfg(feature = "rkyv")]
    pub fn from_bytes(bytes: &'a [u8]) -> AlignedArchivedImmutableKdTree<'a, A, T, K, B> {
        let tree_rk = unsafe { rkyv::archived_root::<ImmutableKdTreeRK<A, T, K, B>>(bytes) };

        AlignedArchivedImmutableKdTree::new_from(tree_rk)
    }

    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }
}

#[cfg(feature = "rkyv")]
impl<A, T, const K: usize, const B: usize> AlignedArchivedImmutableKdTree<'_, A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K> + rkyv::Archive<Archived = A>,
    T: Content + rkyv::Archive<Archived = T>,
    usize: Cast<T>,
{
    /// Returns the current number of elements stored in the tree
    #[inline]
    pub fn size(&self) -> usize {
        self.leaf_items.len()
    }

    /// Returns a LeafSlice for a given leaf index
    #[inline]
    pub(crate) fn get_leaf_slice(&self, leaf_idx: usize) -> LeafSlice<A, T, K> {
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

        // Artificially extend size to be at least chunk length for faster processing
        // TODO: why does this slow things down?
        // let end = end.max(start + 32).min(self.leaf_items.len() as u32);

        LeafSlice::new(
            array_init::array_init(|i| &self.leaf_points[i][start as usize..end as usize]),
            &self.leaf_items[start as usize..end as usize],
        )
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize> From<&[[A; K]]>
    for ImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = (&*points).into();
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    fn from(slice: &[[A; K]]) -> Self {
        ImmutableKdTree::new_from_slice(slice)
    }
}

// prevent clippy complaining that the feature unreliable_select_nth_unstable
// is not defined (I don't want to explicitly define it as if I do then
// passing --all-features in CI will enable it, which I don't want to do
#[allow(unexpected_cfgs)]
impl<A, T, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    /// Creates an `ImmutableKdTree`, balanced and optimized, populated
    /// with items from `source`.
    ///
    /// `ImmutableKdTree` instances are optimally
    /// balanced and tuned, but are not modifiable after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
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

        #[cfg(not(feature = "modified_van_emde_boas"))]
        let stem_node_count = if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two()
        };

        #[cfg(feature = "modified_van_emde_boas")]
        let stem_node_count = if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two() - 1
        };

        let max_stem_level: isize = leaf_node_count.next_power_of_two().ilog2() as isize - 1;

        // TODO: It would be nice to be able to determine the exact required length up-front.
        //  Instead, we just trim the stems afterwards by traversing right-child non-inf nodes
        //  till we hit max level to get the max used stem
        #[cfg(feature = "modified_van_emde_boas")]
        let stem_node_count = stem_node_count * 5;

        let mut stems = avec![A::infinity(); stem_node_count];
        let mut leaf_points: [Vec<A>; K] = array_init(|_| Vec::with_capacity(item_count));
        let mut leaf_items: Vec<T> = Vec::with_capacity(item_count);
        let mut leaf_extents: Vec<(u32, u32)> = Vec::with_capacity(item_count.div_ceil(B));

        let mut sort_index = Vec::from_iter(0..item_count);

        if stem_node_count == 0 {
            // Write leaf and terminate recursion
            leaf_extents.push((0u32, sort_index.len() as u32));

            (0..sort_index.len()).for_each(|i| {
                (0..K).for_each(|dim| leaf_points[dim].push(source[sort_index[i]][dim]));
                leaf_items.push(sort_index[i].az::<T>())
            });
        } else {
            #[cfg(not(feature = "modified_van_emde_boas"))]
            let initial_stem_idx = 1;
            #[cfg(feature = "modified_van_emde_boas")]
            let initial_stem_idx = 0;

            Self::populate_recursive(
                &mut stems,
                0,
                source,
                &mut sort_index,
                initial_stem_idx,
                0,
                max_stem_level,
                leaf_node_count * B,
                &mut leaf_points,
                &mut leaf_items,
                &mut leaf_extents,
            );

            // trim unneeded stems
            #[cfg(feature = "modified_van_emde_boas")]
            if !stems.is_empty() {
                let mut level = 0;
                let mut stem_idx = 0;
                loop {
                    let val = stems[stem_idx];
                    let is_right_child = val.is_finite();
                    stem_idx = modified_van_emde_boas_get_child_idx_v2_branchless(
                        stem_idx,
                        is_right_child,
                        level,
                    );
                    level += 1;
                    if level == max_stem_level as usize {
                        break;
                    }
                }
                stems.truncate(stem_idx + 1);
            }
        }

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
        stems: &mut AVec<A, ConstAlign<128>>,
        dim: usize,
        source: &[[A; K]],
        sort_index: &mut [usize],
        stem_index: usize,
        mut level: usize,
        max_stem_level: isize,
        capacity: usize,
        leaf_points: &mut [Vec<A>; K],
        leaf_items: &mut Vec<T>,
        leaf_extents: &mut Vec<(u32, u32)>,
    ) {
        let chunk_length = sort_index.len();

        if level as isize > max_stem_level {
            // Write leaf and terminate recursion
            leaf_extents.push((
                leaf_items.len() as u32,
                (leaf_items.len() + chunk_length) as u32,
            ));

            (0..chunk_length).for_each(|i| {
                (0..K).for_each(|dim| leaf_points[dim].push(source[sort_index[i]][dim]));
                leaf_items.push(sort_index[i].az::<T>())
            });

            return;
        }

        let levels_below = max_stem_level - level as isize;
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

        #[cfg(feature = "modified_van_emde_boas")]
        let left_child_idx =
            modified_van_emde_boas_get_child_idx_v2_branchless(stem_index, false, level);
        #[cfg(feature = "modified_van_emde_boas")]
        let right_child_idx =
            modified_van_emde_boas_get_child_idx_v2_branchless(stem_index, true, level);

        #[cfg(not(feature = "modified_van_emde_boas"))]
        let left_child_idx = stem_index << 1;
        #[cfg(not(feature = "modified_van_emde_boas"))]
        let right_child_idx = (stem_index << 1) + 1;

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
            return pivot;
        }

        // if the pivot straddles two values that are equal, keep nudging it left until they aren't
        while source[sort_index[pivot]][dim] == source[sort_index[pivot - 1]][dim] && pivot > 1 {
            pivot -= 1;
        }

        pivot
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    ///
    /// let points: Vec<[f64; 3]> = vec!([1.0f64, 2.0f64, 3.0f64]);
    /// let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&points);
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

    /// Returns a LeafSlice for a given leaf index
    #[inline]
    pub(crate) fn get_leaf_slice(&self, leaf_idx: usize) -> LeafSlice<A, T, K> {
        let (start, end) = unsafe { *self.leaf_extents.get_unchecked(leaf_idx) };

        // Artificially extend size to be at least chunk length for faster processing
        // TODO: why does this slow things down?
        // let end = end.max(start + 32).min(self.leaf_items.len() as u32);

        LeafSlice::new(
            array_init::array_init(|i| &self.leaf_points[i][start as usize..end as usize]),
            &self.leaf_items[start as usize..end as usize],
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::SquaredEuclidean;
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};

    #[test]
    fn can_construct_an_empty_tree() {
        let tree = ImmutableKdTree::<f64, u32, 3, 32>::new_from_slice(&[]);
        let _result = tree.nearest_one::<SquaredEuclidean>(&[0.; 3]);
    }

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

        let _tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

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

        let _tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
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

        let _tree: ImmutableKdTree<f32, usize, 2, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

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

            let _tree: ImmutableKdTree<f32, usize, 2, 8> =
                ImmutableKdTree::new_from_slice(&content_to_add);
        }
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_0() {
        let tree_size = 18;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        println!("tree: {:?}", tree);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_1() {
        let tree_size = 33;
        let seed = 100045;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_2() {
        let tree_size = 155;
        let seed = 480;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_3() {
        let tree_size = 26; // also 32
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_4() {
        let tree_size = 21;
        let seed = 131851;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_5() {
        let tree_size = 32;
        let seed = 455191;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_6() {
        let tree_size = 56;
        let seed = 450533;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_7() {
        let tree_size = 18;
        let seed = 992063;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_8() {
        let tree_size = 19;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_9() {
        let tree_size = 20;
        let seed = 894771;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_10() {
        let tree_size = 36;
        let seed = 375096;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_bad_example_11() {
        let tree_size = 10000;
        let seed = 257281;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let content_to_add: Vec<[f32; 4]> = (0..tree_size).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
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

        let _tree: ImmutableKdTree<f32, usize, 4, 8> = ImmutableKdTree::new_from_slice(&duped);
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

        let _tree: ImmutableKdTree<f32, usize, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }

    #[test]
    fn can_construct_optimized_tree_large_rand() {
        const TREE_SIZE: usize = 2usize.pow(23); // ~8M

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(493);
        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let _tree: ImmutableKdTree<f32, usize, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);
    }
}
